from gym import spaces
from dataclasses import dataclass
import os
import numpy as np
from habitat.core.logging import logger
from habitat.config.default_structured_configs import (
    ActionConfig,
)

from hydra.core.config_store import ConfigStore
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction,
    BaseVelNonCylinderAction
)
from habitat.tasks.utils import get_angle
import magnum as mn
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
import habitat_sim
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.rearrange.actions.actions import HumanoidJointAction
from habitat.tasks.rearrange.utils import place_agent_at_dist_from_pos
from habitat.articulated_agent_controllers import HumanoidRearrangeController
import random

play_i = 0

# for humanoid
@registry.register_task_action
class OracleNavAction_wopddl(BaseVelAction, HumanoidJointAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base/humanoid joint control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene. The
    config flag motion_type indicates whether the low level action will be a base_velocity or
    a joint control.
    """

    def __init__(self, *args, task, **kwargs):
        config = kwargs["config"]
        self.motion_type = config.motion_control
        if self.motion_type == "base_velocity":
            BaseVelAction.__init__(self, *args, **kwargs)

        elif self.motion_type == "human_joints":
            HumanoidJointAction.__init__(self, *args, **kwargs)
            self.humanoid_controller = self.spec_inst_humanoid_controller( # self.lazy_inst_humanoid_controller(
                task, config
            )

        else:
            raise ValueError("Unrecognized motion type for oracle nav action")

        self._task = task
        if hasattr(task,"pddl_problem"):
            self._poss_entities = (
                self._task.pddl_problem.get_ordered_entities_list()
            )
        else:
            self._poss_entities = None
        self._prev_ep_id = None
        self.skill_done = False
        self._targets = {}

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        is_left = np.cross(robot_forward, rel) > 0
        if is_left:
            vel = [0, -turn_vel]
        else:
            vel = [0, turn_vel]
        return vel

    def lazy_inst_humanoid_controller(self, task, config):
        # Lazy instantiation of humanoid controller
        # We assign the task with the humanoid controller, so that multiple actions can
        # use it.

        if (
            not hasattr(task, "humanoid_controller")
            or task.humanoid_controller is None
        ):
            # Initialize humanoid controller
            agent_name = self._sim.habitat_config.agents_order[
                self._agent_index
            ]
            walk_pose_path = self._sim.habitat_config.agents[
                agent_name
            ].motion_data_path

            humanoid_controller = HumanoidRearrangeController(walk_pose_path)
            humanoid_controller.set_framerate_for_linspeed(
                config["lin_speed"], config["ang_speed"], self._sim.ctrl_freq
            )
            task.humanoid_controller = humanoid_controller
        return task.humanoid_controller
    
    def spec_inst_humanoid_controller(self, task, config):
        # Instantiation of humanoid controller for specific agent
        # Follow the lazy version, but tell each humanoid agent

        # Initialize humanoid controller
        agent_name = self._sim.habitat_config.agents_order[
            self._agent_index
        ]
        walk_pose_path = self._sim.habitat_config.agents[
            agent_name
        ].motion_data_path

        humanoid_controller = HumanoidRearrangeController(walk_pose_path)
        humanoid_controller.set_framerate_for_linspeed(
            config["lin_speed"], config["ang_speed"], self._sim.ctrl_freq
        )

        exec("task.{0}_humanoid_controller = humanoid_controller".format(agent_name))

        return getattr(task, "{0}_humanoid_controller".format(agent_name))

    def _update_controller_to_navmesh(self):
        base_offset = self.cur_articulated_agent.params.base_offset
        prev_query_pos = self.cur_articulated_agent.base_pos
        target_query_pos = (
            self.humanoid_controller.obj_transform_base.translation
            + base_offset
        )

        filtered_query_pos = self._sim.step_filter(
            prev_query_pos, target_query_pos
        )
        fixup = filtered_query_pos - target_query_pos
        self.humanoid_controller.obj_transform_base.translation += fixup

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id
            self.skill_done = False

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            nav_to_obj = self._poss_entities[nav_to_target_idx]
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                True,
                self.cur_articulated_agent,
            )
            if self.motion_type == "human_joints":
                self.humanoid_controller.reset(
                    self.cur_articulated_agent.base_transformation
                )
            self._targets[nav_to_target_idx] = (
                np.array(start_pos),
                np.array(obj_pos),
            )
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = np.array(self.cur_articulated_agent.base_pos)
        
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        if all(np.array_equal(point, path.points[0]) for point in path.points):
            return [agent_pos, point]
        
        return path.points

    def step(self, *args, **kwargs):
        self.skill_done = False
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_action"
        ]

        if nav_to_target_idx <= 0 or self._poss_entities == None or nav_to_target_idx > len(
            self._poss_entities
        ):
            return
        nav_to_target_idx = int(nav_to_target_idx[0]) - 1
        
        final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            nav_to_target_idx
        )
        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(final_nav_targ)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)

        if curr_path_points is None:
            raise Exception
        else:
            # Compute distance and angle to target
            cur_nav_targ = curr_path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            )

            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction_wopddl._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction_wopddl._compute_turn(
                            rel_targ, self._config.turn_velocity, robot_forward
                        )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                BaseVelAction.step(self, *args, **kwargs)
                return

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                        )
                else:
                    self.humanoid_controller.calculate_stop_pose()
                    self.skill_done = True

                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                HumanoidJointAction.step(self, *args, **kwargs)
                return
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )


@registry.register_task_action
class OracleNavObstacleAction(OracleNavAction_wopddl):
    def __init__(self, *args, task, **kwargs):
        OracleNavAction_wopddl.__init__(self, *args, task=task, **kwargs)
        self.human_num = 0
        self.old_human_pos_list = None
        self.rand_human_speed_scale = np.random.uniform(0.8, 1.2)

    def update_rel_targ_obstacle(
        self, rel_targ, new_human_pos, old_human_pos=None
    ):
        if old_human_pos is None or len(old_human_pos) == 0:
            human_velocity_scale = 0.0
        else:
            # take the norm of the distance between old and new human position
            human_velocity_scale = (
                np.linalg.norm(new_human_pos - old_human_pos) / 0.25
            )  # 0.25 is a magic number
            # set a minimum value for the human velocity scale
            human_velocity_scale = max(human_velocity_scale, 0.1)
        
        std = 3.0
        # scale the amplitude by the human velocity
        amp = 1.0 * human_velocity_scale

        # Get the position of the other agents
        other_agent_rel_pos, other_agent_dist = [], []
        curr_agent_T = np.array(
            self.cur_articulated_agent.base_pos
        )[[0, 2]]

        other_agent_rel_pos.append(rel_targ[None, :])
        other_agent_dist.append(0.0)  # dummy value
        rel_pos = new_human_pos - curr_agent_T
        dist_pos = np.linalg.norm(rel_pos, ord=2, axis=-1) # np.linalg.norm(rel_pos)
        # normalized relative vector
        rel_pos = rel_pos / dist_pos[:, np.newaxis]
        # dist_pos = np.squeeze(dist_pos)
        other_agent_dist.extend(dist_pos)
        other_agent_rel_pos.append(-rel_pos) # -rel_pos[None, :]

        rel_pos = np.concatenate(other_agent_rel_pos)
        rel_dist = np.array(other_agent_dist)
        weight = amp * np.exp(-(rel_dist**2) / std)
        weight[0] = 1.0
        weight_norm = weight[:, None] / weight.sum()
        
        # weighted sum of the old target position and
        # relative position that avoids human
        final_rel_pos = (rel_pos * weight_norm).sum(0)
        return final_rel_pos


    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_obstacle_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def step(self, *args, **kwargs):
        self.skill_done = False
        nav_to_target_coord = kwargs.get(
            self._action_arg_prefix + "oracle_nav_obstacle_action"
        )
        
        if nav_to_target_coord is None or np.linalg.norm(nav_to_target_coord) == 0:
            return None
        self.humanoid_controller.reset(
            self.cur_articulated_agent.base_transformation
        )

        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(nav_to_target_coord)
        current_human_pos = np.array(self.cur_articulated_agent.base_pos)

        if curr_path_points is None:
            raise Exception
        else:
            # Compute distance and angle to target
            if len(curr_path_points) == 1:
                curr_path_points += curr_path_points

            cur_nav_targ = np.array(curr_path_points[1])
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - current_human_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (nav_to_target_coord - current_human_pos)[[0, 2]]

            # NEW: We will update the rel_targ position to avoid the humanoid
            # rel_targ is the next position that the agent wants to walk to
            # old_rel_targ = rel_targ.copy()
            new_human_pos_list = []
            nearby_human_idx = []
            old_rel_targ = rel_targ
            
            if self.human_num > 0:
                # This is very specific to SIRo. Careful merging
                for agent_index in range(self._sim.num_articulated_agents):
                    new_human_pos = np.array(self._sim.agents_mgr[agent_index].articulated_agent.base_pos)
                    new_human_pos_list.append(new_human_pos)
                    if self._agent_index != agent_index and agent_index != 1: # agent_index is actual index, dog is 1, humanoid are 0, 2-8
                        distance = self._sim.geodesic_distance(current_human_pos, new_human_pos)
                        if distance < 2.0 and robot_human_vec_dot_product(new_human_pos, current_human_pos, base_T) > 0.5:
                            nearby_human_idx.append(agent_index)
                
                if self.old_human_pos_list is not None and len(nearby_human_idx) > 0:
                    new_human_pos_array = np.array(new_human_pos_list)
                    old_human_pos_array = np.array(self.old_human_pos_list)
                    rel_targ = self.update_rel_targ_obstacle(
                        rel_targ, new_human_pos_array[nearby_human_idx][:, [0, 2]], old_human_pos_array[nearby_human_idx][:, [0, 2]]
                    )
                self.old_human_pos_list = new_human_pos_list 
            # NEW: If avoiding the human makes us change dir, we will
            # go backwards at times to avoid rotating
                
            dot_prod_rel_targ = (rel_targ * old_rel_targ).sum()
            did_change_dir = dot_prod_rel_targ < 0

            angle_to_target = get_angle(robot_forward, rel_targ) # next goal
            angle_to_final_goal = get_angle(robot_forward, rel_pos) # final goal

            dist_to_final_nav_targ = np.linalg.norm(
                (nav_to_target_coord - current_human_pos)[[0, 2]]
            )
            
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_final_goal < self._config.turn_thresh
            ) or dist_to_final_nav_targ < self._config.dist_thresh / 10.0
            
            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction_wopddl._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        if did_change_dir:
                            if (np.pi - angle_to_target) < self._config.turn_thresh:
                                # Move towards the target
                                vel = [-self._config.forward_velocity, 0]
                            else:
                                vel = OracleNavAction_wopddl._compute_turn(
                                    -rel_targ,
                                    self._config.turn_velocity,
                                    robot_forward,
                                )
                        else:
                            vel = OracleNavAction_wopddl._compute_turn(
                                rel_targ, self._config.turn_velocity, robot_forward
                            )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelAction.step(
                    self, *args, **kwargs
                )
            elif self.motion_type == "human_joints":
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    # Move towards the target
                    if self._config["lin_speed"] == 0:
                        distance_multiplier = 0.0
                    else:
                        distance_multiplier = 1.0  * self.rand_human_speed_scale
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]]),
                            distance_multiplier
                        )
                    else:
                        # Look at the target waypoint.
                        if did_change_dir:
                            if (np.pi - angle_to_target) < self._config.turn_thresh:
                                self.humanoid_controller.calculate_walk_pose(
                                    mn.Vector3([-rel_targ[0], 0.0, -rel_targ[1]]),
                                    distance_multiplier
                                )
                            else:
                                self.humanoid_controller.calculate_walk_pose(
                                    mn.Vector3([-rel_targ[0], 0.0, -rel_targ[1]]),
                                    distance_multiplier
                                )
                        else:
                            self.humanoid_controller.calculate_walk_pose( # turn
                                mn.Vector3([rel_targ[0], 0.0, rel_targ[1]]),
                                distance_multiplier
                            )
                else:
                    self.humanoid_controller.calculate_stop_pose()
                    self.skill_done = True
                    
                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(self, *args, **kwargs)
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav obstacle action"
                )


@registry.register_task_action
class OracleNavRandCoordActionForOtherHuman(OracleNavObstacleAction):  # type: ignore # (OracleNavObstacleAction)
    """
    Oracle Nav RandCoord Action. Selects a random position in the scene and navigates
    there until reaching. When the arg is 1, does replanning.
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self._config = kwargs["config"]
        self.human_num = 4
        self.num_goals = 1
        self.current_goal_idx = 0
        self.wait_step_for_robot = 0
        self.goals = [np.array([0, 0, 0], dtype=np.float32) for _ in range(self.num_goals)]
        # self._largest_indoor_island_idx = get_largest_island_index(
        #     self._sim.pathfinder, self._sim, allow_outdoor=False # True
        # )

    
    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_randcoord_action_obstacle": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _add_n_coord_nav_goals(self,n=3):
        max_tries = 1000
        search_count = 0

        for i in range(n):
            temp_coord_nav = self._sim.pathfinder.get_random_navigable_point(
                max_tries,
                island_index=self._largest_indoor_island_idx,
            )
            while len(self.goals) >= 1 and np.linalg.norm(temp_coord_nav - self.goals[-1],ord=2,axis=-1) < 3:
                temp_coord_nav = self._sim.pathfinder.get_random_navigable_point(
                max_tries,
                island_index=self._largest_indoor_island_idx,)
                search_count += 1
                if search_count > 1000:
                    break

            self.goals[i] = temp_coord_nav

    def reset(self, *args, **kwargs):
        self.human_num = kwargs['task']._human_num
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._prev_ep_id = self._task._episode_id
        self.skill_done = False
        self.coord_nav = None
        self.num_goals = kwargs["episode"].info["num_goals_for_other_human"]
        self.goals = [np.array([0, 0, 0], dtype=np.float32) for _ in range(self.num_goals)]
        self.current_goal_idx = 0
        # self._largest_indoor_island_idx = get_largest_island_index(
        #     self._sim.pathfinder, self._sim, allow_outdoor=False # True
        # )
        if self._agent_index <= self.human_num + 1:
            if self._task._use_episode_start_goal:
                for i in range(self.num_goals):
                    attribute_name = f"human_{self._agent_index - 1}_waypoint_{i+1}_position"
                    if attribute_name in kwargs["episode"].info:
                        self.goals[i] = kwargs["episode"].info[attribute_name]
                    else:
                        self.goals = None
            else:
                self._add_n_coord_nav_goals(self.num_goals)

    def _find_path_given_start_end(self, start, end):
        """Helper function to find the path given starting and end locations"""
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [start, end]
        return path.points

    def _angle_between_vectors(self, v1, v2):
        v1_2d = np.array([v1[0], v1[2]])
        v2_2d = np.array([v2[0], v2[2]])
        
        dot_product = np.dot(v1_2d, v2_2d)
        cos_theta = dot_product / (np.linalg.norm(v1_2d) * np.linalg.norm(v2_2d))
        
        return cos_theta > 0.8

    def _inside_two_agents(self, human_pos, robot_pos, my_pos):
        angle_1 = self._angle_between_vectors(robot_pos - human_pos, robot_pos - my_pos)
        angle_2 = self._angle_between_vectors(human_pos - robot_pos, human_pos - my_pos)

        return angle_1 and angle_2

    def _face_to_agent(self, target_pos, my_pos, base_T):
        """Check if the agent reaches the target agent or not"""
        facing = (
            robot_human_vec_dot_product(target_pos, my_pos, base_T) > 0.5
        )

        return facing

    def _reach_agent(self, target_pos, my_pos):
        """Check if the agent reaches the target agent or not"""

        dis = np.linalg.norm(target_pos - my_pos)

        return dis <= 5.0

    def _get_target_for_coord(self, obj_pos):
        start_pos = obj_pos
        if self.motion_type == "human_joints":
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
        return (start_pos, np.array(obj_pos))

    def step(self, *args, **kwargs):
        self.skill_done = False

        if self.coord_nav is None and self.goals is not None:
            if self.current_goal_idx < len(self.goals):
                self.coord_nav = self.goals[self.current_goal_idx]
                self.current_goal_idx += 1

        kwargs[
            self._action_arg_prefix + "oracle_nav_obstacle_action"
        ] = self.coord_nav

        ret_val = super().step(*args, **kwargs)

        if self.skill_done:
            self.coord_nav = None

        if self._config.human_stop_and_walk_to_robot_distance_threshold != -1:
            robot_id = 1
            robot_pos = self._sim.get_agent_data(
                robot_id
            ).articulated_agent.base_pos

            main_human_id = 0
            main_human_pos = self._sim.get_agent_data(
                main_human_id
            ).articulated_agent.base_pos

            human_pos = self.cur_articulated_agent.base_pos
            
            # The human needs to stop and wait for robot or the main human to move close
            if self._reach_agent(robot_pos, human_pos) or self._reach_agent(main_human_pos, human_pos):
                speed = np.random.uniform(
                    self._config.lin_speed / 2.0, self._config.lin_speed
                )
                lin_speed = speed 
                ang_speed = self._config.ang_speed
                self.humanoid_controller.set_framerate_for_linspeed(
                    lin_speed, ang_speed, self._sim.ctrl_freq
                )
            # Wait for main human and robot
            else:
                self.humanoid_controller.set_framerate_for_linspeed(
                    0, 0, self._sim.ctrl_freq
                )
            
        return ret_val

"""
# @registry.register_task_action
# class OracleNavObstacleActionForMainHuman(OracleNavAction_wopddl):
#     def __init__(self, *args, task, **kwargs):
#         OracleNavAction_wopddl.__init__(self, *args, task=task, **kwargs)
#         self.human_num = 0
#         self.old_human_pos_list = None
#         self.rand_human_speed_scale = np.random.uniform(0.8, 1.2)

#     def update_rel_targ_obstacle(
#         self, rel_targ, new_human_pos, old_human_pos=None
#     ):
#         if old_human_pos is None or len(old_human_pos) == 0:
#             human_velocity_scale = 0.0
#         else:
#             # take the norm of the distance between old and new human position
#             human_velocity_scale = (
#                 np.linalg.norm(new_human_pos - old_human_pos) / 0.25
#             )  # 0.25 is a magic number
#             # set a minimum value for the human velocity scale
#             human_velocity_scale = max(human_velocity_scale, 0.1)

#         std = 8.0
#         # scale the amplitude by the human velocity
#         amp = 8.0 * human_velocity_scale

#         # Get the position of the other agents
#         other_agent_rel_pos, other_agent_dist = [], []
#         curr_agent_T = np.array(
#             self.cur_articulated_agent.base_transformation.translation
#         )[[0, 2]]

#         other_agent_rel_pos.append(rel_targ[None, :])
#         other_agent_dist.append(0.0)  # dummy value
#         rel_pos = new_human_pos - curr_agent_T
#         dist_pos = np.linalg.norm(rel_pos, ord=2, axis=-1) # np.linalg.norm(rel_pos)
#         # normalized relative vector
#         rel_pos = rel_pos / dist_pos[:, np.newaxis]
#         # dist_pos = np.squeeze(dist_pos)
#         other_agent_dist.extend(dist_pos)
#         other_agent_rel_pos.append(-rel_pos) # -rel_pos[None, :]

#         rel_pos = np.concatenate(other_agent_rel_pos)
#         rel_dist = np.array(other_agent_dist)
#         weight = amp * np.exp(-(rel_dist**2) / std)
#         weight[0] = 1.0
#         # TODO: explore softmax?
#         weight_norm = weight[:, None] / weight.sum()
#         # weighted sum of the old target position and
#         # relative position that avoids human
#         final_rel_pos = (rel_pos * weight_norm).sum(0)
#         return final_rel_pos

#     @property
#     def action_space(self):
#         return spaces.Dict(
#             {
#                 self._action_arg_prefix
#                 + "oracle_nav_obstacle_action_main_human": spaces.Box(
#                     shape=(1,),
#                     low=np.finfo(np.float32).min,
#                     high=np.finfo(np.float32).max,
#                     dtype=np.float32,
#                 )
#             }
#         )

#     def step(self, *args, **kwargs):
#         self.skill_done = False
#         nav_to_target_coord = kwargs.get(
#             self._action_arg_prefix + "oracle_nav_obstacle_action_main_human"
#         )
        
#         if nav_to_target_coord is None or np.linalg.norm(nav_to_target_coord) == 0:
#             return None
#         self.humanoid_controller.reset(
#                 self.cur_articulated_agent.base_transformation
#             )

#         base_T = self.cur_articulated_agent.base_transformation
#         curr_path_points = self._path_to_point(nav_to_target_coord)
#         current_human_pos = np.array(self.cur_articulated_agent.base_pos)

#         if curr_path_points is None:
#             raise Exception
#         else:
#             # Compute distance and angle to target
#             if len(curr_path_points) == 1:
#                 curr_path_points += curr_path_points
#             cur_nav_targ = curr_path_points[1]
#             forward = np.array([1.0, 0, 0])
#             robot_forward = np.array(base_T.transform_vector(forward))

#             # Compute relative target.
#             rel_targ = cur_nav_targ - current_human_pos

#             # Compute heading angle (2D calculation)
#             robot_forward = robot_forward[[0, 2]]
#             rel_targ = rel_targ[[0, 2]]
#             rel_pos = (nav_to_target_coord - current_human_pos)[[0, 2]]

#             # NEW: We will update the rel_targ position to avoid the humanoid
#             # rel_targ is the next position that the agent wants to walk to
#             # old_rel_targ = rel_targ.copy()
#             new_human_pos_list = []
#             nearby_human_idx = []
#             old_rel_targ = rel_targ

#             if self.human_num > 0:
#                 # This is very specific to SIRo. Careful merging
#                 for agent_index in range(2, self._sim.num_articulated_agents):
#                     new_human_pos = np.array(
#                         self._sim.get_agent_data(
#                             agent_index
#                         ).articulated_agent.base_transformation.translation
#                     )
#                     new_human_pos_list.append(new_human_pos)
#                     if self._agent_index != agent_index: # agent_index is actual index, dog is zero, humanoid are 1-8
#                         distance = self._sim.geodesic_distance(current_human_pos, new_human_pos)
#                         if distance < 3.0 and robot_human_vec_dot_product(new_human_pos, current_human_pos, base_T) > 0.5:
#                             nearby_human_idx.append(agent_index-2)
                
#                 if self.old_human_pos_list is not None and len(nearby_human_idx) > 0:
#                     new_human_pos_array = np.array(new_human_pos_list)
#                     old_human_pos_array = np.array(self.old_human_pos_list)
#                     rel_targ = self.update_rel_targ_obstacle(
#                         rel_targ, new_human_pos_array[nearby_human_idx][:, [0, 2]], old_human_pos_array[nearby_human_idx][:, [0, 2]]
#                     )
#                 self.old_human_pos_list = new_human_pos_list 
#             # NEW: If avoiding the human makes us change dir, we will
#             # go backwards at times to avoid rotating
                
#             dot_prod_rel_targ = (rel_targ * old_rel_targ).sum()
#             did_change_dir = dot_prod_rel_targ < 0

#             angle_to_target = get_angle(robot_forward, rel_targ) # next goal
#             angle_to_obj = get_angle(robot_forward, rel_pos) # final goal

#             dist_to_final_nav_targ = np.linalg.norm(
#                 (nav_to_target_coord - current_human_pos)[[0, 2]]
#             )
#             at_goal = (
#                 dist_to_final_nav_targ < self._config.dist_thresh
#                 and angle_to_obj < self._config.turn_thresh
#             ) or dist_to_final_nav_targ < self._config.dist_thresh / 10.0
#             if self.motion_type == "base_velocity":
#                 if not at_goal:
#                     if dist_to_final_nav_targ < self._config.dist_thresh:
#                         # Look at the object
#                         vel = OracleNavAction_wopddl._compute_turn(
#                             rel_pos, self._config.turn_velocity, robot_forward
#                         )
#                     elif angle_to_target < self._config.turn_thresh:
#                         # Move towards the target
#                         vel = [self._config.forward_velocity, 0]
#                     else:
#                         # Look at the target waypoint.
#                         if did_change_dir:
#                             if (np.pi - angle_to_target) < self._config.turn_thresh:
#                                 # Move towards the target
#                                 vel = [-self._config.forward_velocity, 0]
#                             else:
#                                 vel = OracleNavAction_wopddl._compute_turn(
#                                     -rel_targ,
#                                     self._config.turn_velocity,
#                                     robot_forward,
#                                 )
#                         else:
#                             vel = OracleNavAction_wopddl._compute_turn(
#                                 rel_targ, self._config.turn_velocity, robot_forward
#                             )
#                 else:
#                     vel = [0, 0]
#                     self.skill_done = True
#                 kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
#                 return BaseVelAction.step(
#                     self, *args, **kwargs
#                 )
#             elif self.motion_type == "human_joints":
#                 self.humanoid_controller.obj_transform_base = base_T
#                 if not at_goal:
#                     if dist_to_final_nav_targ < self._config.dist_thresh:
#                         # Look at the object
#                         self.humanoid_controller.calculate_turn_pose(
#                             mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
#                         )
#                     elif angle_to_target < self._config.turn_thresh:
#                         # Move towards the target
#                         if self._config["lin_speed"] == 0:
#                             distance_multiplier = 0.0
#                         else:
#                             distance_multiplier = 1.0  * self.rand_human_speed_scale
#                         self.humanoid_controller.calculate_walk_pose(
#                             mn.Vector3([rel_targ[0], 0.0, rel_targ[1]]),
#                             distance_multiplier
#                             )
#                     else:
#                         # Look at the target waypoint.
#                         if did_change_dir:
#                             if (np.pi - angle_to_target) < self._config.turn_thresh:
#                                 # Move towards the target
#                                 if self._config["lin_speed"] == 0:
#                                     distance_multiplier = 0.0
#                                 else:
#                                     distance_multiplier = 1.0
#                                 self.humanoid_controller.calculate_walk_pose(
#                                 mn.Vector3([-rel_targ[0], 0.0, -rel_targ[1]]),
#                                 distance_multiplier
#                                 )
#                             else:
#                                 self.humanoid_controller.calculate_walk_pose(
#                                     mn.Vector3([-rel_targ[0], 0.0, -rel_targ[1]])
#                                     )
#                         else:
#                             self.humanoid_controller.calculate_walk_pose( # turn
#                                 mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
#                                 )
#                 else:
#                     self.humanoid_controller.calculate_stop_pose()
#                     self.skill_done = True
#                 self._update_controller_to_navmesh()
#                 base_action = self.humanoid_controller.get_pose()
#                 kwargs[
#                     f"{self._action_arg_prefix}human_joints_trans"
#                 ] = base_action

#                 return HumanoidJointAction.step(self, *args, **kwargs)
#             else:
#                 raise ValueError(
#                     "Unrecognized motion type for oracle nav obstacle action"
#                 )
"""           

@registry.register_task_action
class OracleNavCoordActionObstacleForMainHuman(OracleNavObstacleAction):
    """
    Oracle Nav Coord Action for main humnaoid.
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self._config = kwargs["config"]
        self.num_goals = 1
        self.current_goal_idx = 0
        self.wait_step_for_robot = 0
        self.goals = [np.array([0, 0, 0], dtype=np.float32) for _ in range(self.num_goals)]
        
    
    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_obstacle_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.human_num = kwargs['task']._human_num
        if self._task._episode_id != self._prev_ep_id:
            self._prev_ep_id = self._task._episode_id
        self.skill_done = False
        self.coord_nav = None
        self.num_goals = kwargs["episode"].info["num_goals_for_main_human"]
        self.goals = [np.array([0, 0, 0], dtype=np.float32) for _ in range(self.num_goals)]
        self.current_goal_idx = 0
        self.wait_step_for_robot = 0
        kwargs['task'].is_stop_called = False
        # self.max_stop_step = random.randint(5, 15)
        self.max_stop_step = 50
        
        if self._agent_index == 0:
            for goal in kwargs["episode"].goals:
                self.goals = [np.array(pos, dtype=np.float32) for pos in goal.position]
        else:
            raise("OracleNavCoordActionObstacleForMainHuman is only for the main humanoid!")

    def step(self, *args, **kwargs):
        self.skill_done = False
        
        if self.coord_nav is None and self.goals is not None:
            if self.current_goal_idx < len(self.goals):
                self.coord_nav = self.goals[self.current_goal_idx]
                self.current_goal_idx += 1
            else:
                self.wait_step_for_robot += 1
                # print(f"Finished navigation! Current wait step: {self.wait_step_for_robot}")
                if self.wait_step_for_robot > self.max_stop_step:
                    kwargs['task'].should_end = True
                    kwargs['task'].is_stop_called = True

        kwargs[
            self._action_arg_prefix + "oracle_nav_obstacle_action"
        ] = self.coord_nav

        ret_val = super().step(*args, **kwargs)

        if self.skill_done:
            self.coord_nav = None

        lin_speed = np.random.uniform(
            self._config.lin_speed / 3.0, self._config.lin_speed * 1.2
        )
        ang_speed = self._config.ang_speed
        self.humanoid_controller.set_framerate_for_linspeed(
            lin_speed, ang_speed, self._sim.ctrl_freq
        )
        
        return ret_val


# for robot
@registry.register_task_action
class OracleNavCoordinateActionForRobot(BaseVelNonCylinderAction):  # type: ignore
    """
    An action to drive the robot agent to the main human
    """

    def __init__(self, *args, task, **kwargs):
        self._targets = {}
        # Initialize teacher base velocity to zeros so external readers
        # (e.g. baseline_agent.py reading ``last_teacher_base_vel``) never
        # observe ``None`` before the first ``step`` call.
        self.last_teacher_base_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_teacher_base_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_human_xz: Optional[np.ndarray] = None
        self._human_velocity_xz: np.ndarray = np.zeros(2, dtype=np.float32)
        self._last_human_motion_dir: Optional[np.ndarray] = None
        self._smoothed_follow_target_xz: Optional[np.ndarray] = None
        self._cached_path_points: Optional[List[np.ndarray]] = None
        self._cached_path_goal: Optional[np.ndarray] = None
        self._path_steps_since_plan: int = 0
        BaseVelNonCylinderAction.__init__(self, *args, **kwargs)

    def _teacher_max_forward_speed(self) -> float:
        return float(
            getattr(
                self._config,
                "longitudinal_lin_speed",
                getattr(self._config, "max_forward_speed", 2.0),
            )
        )

    def _teacher_max_tangent_speed(self) -> float:
        return float(
            getattr(
                self._config,
                "lateral_lin_speed",
                getattr(self._config, "max_tangent_speed", 1.0),
            )
        )

    def _teacher_max_yaw_speed(self) -> float:
        return float(
            getattr(
                self._config,
                "ang_speed",
                getattr(self._config, "max_yaw_speed", 2.5),
            )
        )

    @property
    def action_space(self):
        # Placeholder: this oracle action is fully self-driven (target taken
        # from the main human inside ``step``) and consumes no policy input,
        # so it intentionally exposes an empty action space.
        return spaces.Dict({})

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        # Clear per-episode spawn cache and teacher velocity to avoid leaking
        # state across episodes when the action instance is reused.
        self._targets = {}
        self.last_teacher_base_vel = np.zeros(3, dtype=np.float32)
        self._prev_teacher_base_vel = np.zeros(3, dtype=np.float32)
        self._prev_human_xz = None
        self._human_velocity_xz = np.zeros(2, dtype=np.float32)
        self._last_human_motion_dir = None
        self._smoothed_follow_target_xz = None
        self._cached_path_points = None
        self._cached_path_goal = None
        self._path_steps_since_plan = 0

    def _get_target_for_coord(self, look_at_pos, agent_pos):
        """Given a place to look at, selects an agent_pos to navigate.

        ``precision`` controls how aggressively the spawn-point cache is
        re-used across calls. The previous value of 0.25m was large enough
        that a walking human could move ~25cm within a single quantization
        cell before the cache key changed, which kept the robot locked on a
        stale spawn point and caused the "spin in place while the human walks
        away" failure observed during data collection. 0.05m keeps the cache
        useful for genuinely-stationary humans (multiple consecutive frames
        within 5cm) while making the spawn point follow any real motion at
        roughly the simulator's per-frame resolution.
        """
        precision = 0.05
        pos_key = np.around(look_at_pos / precision, decimals=0) * precision
        pos_key = tuple(pos_key)

        if pos_key not in self._targets:
            agent_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(look_at_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                True,
                self.cur_articulated_agent,
                navmesh_offset=getattr(self._config, "navmesh_offset", None),
            )
            self._targets[pos_key] = agent_pos
        else:
            agent_pos = self._targets[pos_key]

        return (agent_pos, np.array(look_at_pos))


    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def _teacher_ped_repulse_enabled(self) -> bool:
        # RVO2 is the default dynamic-obstacle controller. Keeping a second
        # potential-field controller enabled at the same time made the two
        # layers fight each other and produced lateral/yaw chatter.
        return os.environ.get("TEACHER_PED_REPULSE", "0") != "0"

    def _teacher_ped_repulse_radius(self) -> float:
        try:
            return float(os.environ.get("TEACHER_PED_REPULSE_RADIUS", "1.5"))
        except ValueError:
            return 1.5

    def _teacher_ped_repulse_gain(self) -> float:
        try:
            return float(os.environ.get("TEACHER_PED_REPULSE_GAIN", "0.8"))
        except ValueError:
            return 0.8

    def _other_humanoid_xz_positions(self) -> List[np.ndarray]:
        """Other pedestrians on XZ (exclude main human agent_0 and robot agent_1)."""
        positions: List[np.ndarray] = []
        for idx in range(self._sim.num_articulated_agents):
            if idx in (0, 1):
                continue
            pos = np.array(
                self._sim.agents_mgr[idx].articulated_agent.base_pos,
                dtype=np.float64,
            )
            positions.append(pos[[0, 2]])
        return positions

    def _repulse_toward_main(
        self, rel_xz: np.ndarray, robot_pos: np.ndarray
    ) -> np.ndarray:
        """Repulse rel_xz away from nearby other humanoids while keeping main-target bias."""
        if not self._teacher_ped_repulse_enabled():
            return rel_xz

        rel = np.asarray(rel_xz, dtype=np.float64).reshape(2)
        rel_norm = float(np.linalg.norm(rel))
        if rel_norm < 1e-6:
            return rel_xz

        robot_xz = np.asarray(
            [float(robot_pos[0]), float(robot_pos[2])], dtype=np.float64
        )
        radius = self._teacher_ped_repulse_radius()
        gain = self._teacher_ped_repulse_gain()

        repulse = np.zeros(2, dtype=np.float64)
        for other_xz in self._other_humanoid_xz_positions():
            delta = other_xz - robot_xz
            dist = float(np.linalg.norm(delta))
            if dist >= radius or dist < 1e-4:
                continue
            away = (robot_xz - other_xz) / dist
            weight = gain * (1.0 - dist / radius)
            repulse += weight * away

        if float(np.linalg.norm(repulse)) < 1e-6:
            return rel_xz

        combined = rel / rel_norm + repulse
        combined_norm = float(np.linalg.norm(combined))
        if combined_norm < 1e-6:
            return rel_xz
        return (combined / combined_norm * rel_norm).astype(np.float32)

    def _update_human_velocity(self, human_xz: np.ndarray) -> np.ndarray:
        """Estimate a smooth main-human velocity in world XZ coordinates."""
        human_xz = np.asarray(human_xz, dtype=np.float64).reshape(2)
        dt = 1.0 / max(float(self._sim.ctrl_freq), 1.0)
        if self._prev_human_xz is None:
            raw_velocity = np.zeros(2, dtype=np.float64)
        else:
            raw_velocity = (human_xz - self._prev_human_xz) / dt

        max_human_speed = max(
            0.1, float(getattr(self._config, "max_tracked_human_speed", 2.5))
        )
        raw_speed = float(np.linalg.norm(raw_velocity))
        if raw_speed > max_human_speed:
            raw_velocity *= max_human_speed / raw_speed

        alpha = float(
            np.clip(
                getattr(self._config, "human_velocity_ema_alpha", 0.35),
                0.0,
                1.0,
            )
        )
        self._human_velocity_xz = (
            (1.0 - alpha) * self._human_velocity_xz + alpha * raw_velocity
        ).astype(np.float32)
        self._prev_human_xz = human_xz.copy()

        speed = float(np.linalg.norm(self._human_velocity_xz))
        min_motion_speed = max(
            0.0, float(getattr(self._config, "min_human_motion_speed", 0.08))
        )
        if speed > min_motion_speed:
            self._last_human_motion_dir = (
                self._human_velocity_xz / speed
            ).astype(np.float32)
        return self._human_velocity_xz.copy()

    def _follow_target_xz(
        self,
        human_xz: np.ndarray,
        robot_xz: np.ndarray,
    ) -> np.ndarray:
        """Return a temporally smooth target behind the moving main human."""
        human_xz = np.asarray(human_xz, dtype=np.float64).reshape(2)
        robot_xz = np.asarray(robot_xz, dtype=np.float64).reshape(2)

        motion_dir = self._last_human_motion_dir
        if motion_dir is None:
            robot_to_human = human_xz - robot_xz
            distance = float(np.linalg.norm(robot_to_human))
            if distance > 1e-6:
                motion_dir = robot_to_human / distance
            else:
                motion_dir = np.array([1.0, 0.0], dtype=np.float64)

        desired_distance = max(0.1, float(self._config.dist_thresh))
        raw_target = human_xz - desired_distance * np.asarray(motion_dir)
        alpha = float(
            np.clip(
                getattr(self._config, "follow_target_ema_alpha", 0.4),
                0.0,
                1.0,
            )
        )
        if self._smoothed_follow_target_xz is None:
            smoothed = raw_target
        else:
            smoothed = (
                (1.0 - alpha) * self._smoothed_follow_target_xz
                + alpha * raw_target
            )
        self._smoothed_follow_target_xz = np.asarray(
            smoothed, dtype=np.float32
        )
        return self._smoothed_follow_target_xz.copy()

    @staticmethod
    def _path_lookahead_xz(
        path_points: List[np.ndarray],
        robot_xz: np.ndarray,
        lookahead_distance: float,
    ) -> Tuple[np.ndarray, float]:
        """Project onto a path and return a lookahead point plus remaining length."""
        points = [np.asarray(point, dtype=np.float64)[[0, 2]] for point in path_points]
        robot_xz = np.asarray(robot_xz, dtype=np.float64).reshape(2)
        if len(points) == 0:
            return robot_xz.copy(), 0.0
        if len(points) == 1:
            return points[0].copy(), float(np.linalg.norm(points[0] - robot_xz))

        best_dist = float("inf")
        best_idx = 0
        best_point = points[0]
        for idx in range(len(points) - 1):
            start, end = points[idx], points[idx + 1]
            segment = end - start
            segment_len_sq = float(np.dot(segment, segment))
            if segment_len_sq < 1e-12:
                t = 0.0
                projected = start
            else:
                t = float(
                    np.clip(
                        np.dot(robot_xz - start, segment) / segment_len_sq,
                        0.0,
                        1.0,
                    )
                )
                projected = start + t * segment
            distance = float(np.linalg.norm(robot_xz - projected))
            if distance < best_dist:
                best_dist = distance
                best_idx = idx
                best_point = projected

        remaining_length = float(np.linalg.norm(points[best_idx + 1] - best_point))
        for idx in range(best_idx + 1, len(points) - 1):
            remaining_length += float(np.linalg.norm(points[idx + 1] - points[idx]))

        remaining_lookahead = max(0.0, float(lookahead_distance))
        current = best_point
        for idx in range(best_idx, len(points) - 1):
            end = points[idx + 1]
            segment = end - current
            segment_length = float(np.linalg.norm(segment))
            if segment_length >= remaining_lookahead and segment_length > 1e-9:
                return (
                    current + segment * (remaining_lookahead / segment_length),
                    remaining_length,
                )
            remaining_lookahead -= segment_length
            current = end

        return points[-1].copy(), remaining_length

    def _path_for_follow_target(self, goal: np.ndarray) -> List[np.ndarray]:
        """Re-use a short-lived path so adjacent teacher commands share geometry."""
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        replan_interval = max(
            1, int(getattr(self._config, "path_replan_interval", 3))
        )
        replan_distance = max(
            0.01, float(getattr(self._config, "path_replan_distance", 0.25))
        )
        should_replan = (
            self._cached_path_points is None
            or self._cached_path_goal is None
            or self._path_steps_since_plan >= replan_interval
            or float(np.linalg.norm((goal - self._cached_path_goal)[[0, 2]]))
            >= replan_distance
        )
        if should_replan:
            path_points = self._path_to_point(goal)
            if path_points is None or len(path_points) == 0:
                path_points = [
                    np.asarray(self.cur_articulated_agent.base_pos),
                    goal,
                ]
            self._cached_path_points = [np.asarray(point) for point in path_points]
            self._cached_path_goal = goal.copy()
            self._path_steps_since_plan = 0
        else:
            self._path_steps_since_plan += 1
        return self._cached_path_points

    def _normalized_velocity_from_world(
        self,
        velocity_world_xz: np.ndarray,
        robot_forward: np.ndarray,
        yaw_velocity: float,
    ) -> np.ndarray:
        """Convert physical world-XZ velocity to normalized BaseVel action."""
        eps = 1e-8
        robot_forward = np.asarray(robot_forward, dtype=np.float64).reshape(2)
        robot_forward /= float(np.linalg.norm(robot_forward)) + eps
        robot_plus_z = np.array([-robot_forward[1], robot_forward[0]])
        velocity_world_xz = np.asarray(velocity_world_xz, dtype=np.float64).reshape(2)

        forward_speed = float(np.dot(velocity_world_xz, robot_forward))
        lateral_speed = -float(np.dot(velocity_world_xz, robot_plus_z))
        max_forward = max(self._teacher_max_forward_speed(), eps)
        max_lateral = max(self._teacher_max_tangent_speed(), eps)
        max_yaw = max(self._teacher_max_yaw_speed(), eps)

        # Preserve direction while projecting the holonomic command into the
        # executor's anisotropic speed envelope.
        scale = max(
            1.0,
            abs(forward_speed) / max_forward,
            abs(lateral_speed) / max_lateral,
        )
        forward_speed /= scale
        lateral_speed /= scale
        yaw_velocity = float(np.clip(yaw_velocity, -max_yaw, max_yaw))
        return np.asarray(
            [
                forward_speed / max_forward,
                lateral_speed / max_lateral,
                yaw_velocity / max_yaw,
            ],
            dtype=np.float32,
        )

    def _slew_limit_teacher(self, desired: np.ndarray) -> np.ndarray:
        """Limit physical acceleration before exposing normalized teacher actions."""
        desired = np.clip(np.asarray(desired, dtype=np.float64).reshape(3), -1.0, 1.0)
        previous = np.asarray(self._prev_teacher_base_vel, dtype=np.float64)
        speed_caps = np.asarray(
            [
                self._teacher_max_forward_speed(),
                self._teacher_max_tangent_speed(),
                self._teacher_max_yaw_speed(),
            ],
            dtype=np.float64,
        )
        accel_caps = np.asarray(
            [
                max(0.0, float(getattr(self._config, "max_forward_accel", 3.0))),
                max(0.0, float(getattr(self._config, "max_lateral_accel", 2.5))),
                max(0.0, float(getattr(self._config, "max_yaw_accel", 6.0))),
            ],
            dtype=np.float64,
        )
        dt = 1.0 / max(float(self._sim.ctrl_freq), 1.0)
        previous_physical = previous * speed_caps
        desired_physical = desired * speed_caps
        max_delta = accel_caps * dt
        limited_physical = previous_physical + np.clip(
            desired_physical - previous_physical,
            -max_delta,
            max_delta,
        )
        limited = np.divide(
            limited_physical,
            np.maximum(speed_caps, 1e-8),
        )
        if not bool(getattr(self._config, "allow_back", True)):
            limited[0] = max(0.0, limited[0])
        limited = np.clip(limited, -1.0, 1.0).astype(np.float32)
        self._prev_teacher_base_vel = limited.copy()
        self.last_teacher_base_vel = limited.copy()
        return limited

    
    def compute_teacher_base_vel(self) -> np.ndarray:
        """Smooth continuous follower velocity without applying it to the robot."""
        main_human_id = 0
        nav_to_target_coord = self._sim.get_agent_data(
            main_human_id
        ).articulated_agent.base_pos
        nav_position_coord = None

        if nav_to_target_coord is None or not np.all(
            np.isfinite(np.asarray(nav_to_target_coord, dtype=np.float32))
        ):
            return self._slew_limit_teacher(np.zeros(3, dtype=np.float32))

        base_T = self.cur_articulated_agent.base_transformation
        robot_pos = np.array(self.cur_articulated_agent.base_pos)
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(base_T.transform_vector(forward))
        robot_forward = robot_forward[[0, 2]]
        human_pos = np.asarray(nav_to_target_coord, dtype=np.float64)
        human_xz = human_pos[[0, 2]]
        robot_xz = robot_pos[[0, 2]]
        human_velocity = self._update_human_velocity(human_xz)

        follow_target_xz = self._follow_target_xz(human_xz, robot_xz)
        follow_target_3d = np.asarray(
            [follow_target_xz[0], human_pos[1], follow_target_xz[1]],
            dtype=np.float64,
        )
        final_nav_targ, _ = self._get_target_for_coord(
            follow_target_3d, nav_position_coord
        )
        path_points = self._path_for_follow_target(np.asarray(final_nav_targ))
        lookahead_distance = max(
            0.05, float(getattr(self._config, "path_lookahead", 0.8))
        )
        lookahead_xz, remaining_path_distance = self._path_lookahead_xz(
            path_points,
            robot_xz,
            lookahead_distance,
        )
        path_vector = np.asarray(lookahead_xz) - robot_xz
        path_vector_norm = float(np.linalg.norm(path_vector))
        if path_vector_norm > 1e-6:
            path_direction = path_vector / path_vector_norm
        else:
            target_vector = follow_target_xz - robot_xz
            target_norm = float(np.linalg.norm(target_vector))
            if target_norm > 1e-6:
                path_direction = target_vector / target_norm
            else:
                human_speed = float(np.linalg.norm(human_velocity))
                if human_speed > 1e-6:
                    path_direction = human_velocity / human_speed
                else:
                    path_direction = robot_forward / (
                        float(np.linalg.norm(robot_forward)) + 1e-8
                    )

        position_gain = max(
            0.0, float(getattr(self._config, "follow_position_gain", 1.4))
        )
        feedforward_gain = max(
            0.0, float(getattr(self._config, "human_velocity_feedforward", 0.9))
        )
        desired_world_velocity = (
            position_gain * remaining_path_distance * path_direction
            + feedforward_gain * human_velocity
        )

        rel_human = human_xz - robot_xz
        distance_to_human = float(np.linalg.norm(rel_human))
        safety_distance = max(
            0.1, float(getattr(self._config, "safety_distance", 0.85))
        )
        if 1e-6 < distance_to_human < safety_distance:
            repulsion_gain = max(
                0.0,
                float(getattr(self._config, "safety_repulsion_gain", 4.0)),
            )
            desired_world_velocity += (
                repulsion_gain
                * (safety_distance - distance_to_human)
                * (-rel_human / distance_to_human)
            )

        desired_world_velocity = self._repulse_toward_main(
            desired_world_velocity, robot_pos
        )

        heading_angle = get_angle(robot_forward, rel_human)
        if np.cross(robot_forward, rel_human) > 0:
            heading_angle = -heading_angle
        yaw_deadband = max(
            0.0, float(getattr(self._config, "yaw_deadband", 0.025))
        )
        if abs(heading_angle) <= yaw_deadband:
            yaw_velocity = 0.0
        else:
            yaw_gain = max(
                0.0, float(getattr(self._config, "follow_yaw_gain", 2.2))
            )
            yaw_velocity = (
                np.sign(heading_angle)
                * yaw_gain
                * (abs(heading_angle) - yaw_deadband)
            )

        desired = self._normalized_velocity_from_world(
            desired_world_velocity,
            robot_forward,
            yaw_velocity,
        )
        return self._slew_limit_teacher(desired)

    def step(self, *args, **kwargs):
        vel = self.compute_teacher_base_vel()
        kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
        return BaseVelNonCylinderAction.step(self, *args, **kwargs)



@dataclass
class DiscreteStopActionConfig(ActionConfig):
    type: str = "DiscreteStopAction"
    lin_speed: float = 0.0
    ang_speed: float = 0.0 
    allow_back: bool = False
    value: int = 1
    allow_dyn_slide: bool = False # True
    leg_animation_checkpoint: str = (
        "data/robots/spot_data/spot_walking_trajectory.csv"
    )
    play_i_perframe: int = 5
    use_range: Optional[List[int]] = field(default_factory=lambda: [107, 863])

@dataclass
class DiscreteMoveForwardActionConfig(ActionConfig):
    type: str = "DiscreteMoveForwardAction"
    lin_speed: float = 10.0
    ang_speed: float = 0.0 
    allow_back: bool = False
    value: int = 1
    allow_dyn_slide: bool = False # True
    leg_animation_checkpoint: str = (
        "data/robots/spot_data/spot_walking_trajectory.csv"
    )
    play_i_perframe: int = 5
    use_range: Optional[List[int]] = field(default_factory=lambda: [107, 863])

@dataclass
class DiscreteTurnLeftActionConfig(ActionConfig):
    type: str = "DiscreteTurnLeftAction"
    lin_speed: float = 0.0
    ang_speed: float = 10.0 
    allow_back: bool = False
    value: int = 1
    allow_dyn_slide: bool = False # True
    leg_animation_checkpoint: str = (
        "data/robots/spot_data/spot_walking_trajectory.csv"
    )
    play_i_perframe: int = 5
    use_range: Optional[List[int]] = field(default_factory=lambda: [107, 863])

@dataclass
class DiscreteTurnRightActionConfig(ActionConfig):
    type: str = "DiscreteTurnRightAction"
    lin_speed: float = 0.0
    ang_speed: float = -10.0 
    allow_back: bool = False
    value: int = 1
    allow_dyn_slide: bool = False # True
    leg_animation_checkpoint: str = (
        "data/robots/spot_data/spot_walking_trajectory.csv"
    )
    play_i_perframe: int = 5
    use_range: Optional[List[int]] = field(default_factory=lambda: [107, 863])

@dataclass
class OracleNavActionWOPDDLConfig(ActionConfig):
    """
    Rearrangement Only, Oracle navigation action.
    This action takes as input a discrete ID which refers to an object in the
    PDDL domain. The oracle navigation controller then computes the actions to
    navigate to that desired object.
    """

    type: str = "OracleNavAction_wopddl"
    # Whether the motion is in the form of base_velocity or human_joints
    motion_control: str = "human_joints"
    num_joints: int = 17
    turn_velocity: float = 1.0
    forward_velocity: float = 1.0
    turn_thresh: float = 0.1
    dist_thresh: float = 0.2
    lin_speed: float = 10.0
    ang_speed: float = 10.0
    allow_dyn_slide: bool = True
    allow_back: bool = True
    spawn_max_dist_to_obj: float = 2.0
    num_spawn_attempts: int = 200
    # For social nav training only. It controls the distance threshold
    # between the robot and the human and decide if the human wants to walk or not
    human_stop_and_walk_to_robot_distance_threshold: float = -1.0

@dataclass
class OracleFollowActionConfig(ActionConfig):
    r"""
    In Rearrangement only for the non cylinder shape of the robot. Corresponds to the base velocity. Contains two continuous actions, the first one controls forward and backward motion, the second the rotation.
    """
    type: str = "OracleNavCoordinateActionForRobot"

    # The max longitudinal and lateral linear speeds of the robot (m/s, rad/s).
    lin_speed: float = 2.0
    longitudinal_lin_speed: float = 2.0
    lateral_lin_speed: float = 1.0
    # The max angular speed of the robot
    ang_speed: float = 2.5
    # If we want to do sliding or not
    allow_dyn_slide: bool = False
    # If we allow the robot to move back or not
    allow_back: bool = True
    # There is a collision if the difference between the clamped NavMesh position and target position
    # is more than collision_threshold for any point.
    collision_threshold: float = 1e-5
    # If we allow the robot to move laterally.
    enable_lateral_move: bool = False
    # If the condition of sliding includes the checking of rotation
    enable_rotation_check_for_dyn_slide: bool = True

    # For oracle follower
    turn_thresh: float = 0.1
    dist_thresh: float = 0.2
    spawn_max_dist_to_obj: float = -1
    num_spawn_attempts: int = 200

    # Smooth continuous following controller. ``dist_thresh`` is retained as
    # the desired following distance for backwards-compatible YAML configs.
    follow_position_gain: float = 1.4
    follow_yaw_gain: float = 2.2
    human_velocity_feedforward: float = 0.9
    human_velocity_ema_alpha: float = 0.35
    follow_target_ema_alpha: float = 0.4
    min_human_motion_speed: float = 0.08
    max_tracked_human_speed: float = 2.5
    path_lookahead: float = 0.8
    path_replan_interval: int = 3
    path_replan_distance: float = 0.25
    yaw_deadband: float = 0.025
    safety_distance: float = 0.85
    safety_repulsion_gain: float = 4.0
    max_forward_accel: float = 3.0
    max_lateral_accel: float = 2.5
    max_yaw_accel: float = 6.0

    # Teacher normalization caps; should match longitudinal/lateral/ang_speed in yaml.
    max_forward_speed: float = 2.0
    max_tangent_speed: float = 1.0
    max_yaw_speed: float = 2.5

    # Not used here
    navmesh_offset: Optional[List[float]] = None


cs = ConfigStore.instance()

cs.store( ##
    package="habitat.task.actions.discrete_stop",
    group="habitat/task/actions",
    name="discrete_stop",
    node=DiscreteStopActionConfig,
)

cs.store( ##
    package="habitat.task.actions.discrete_move_forward",
    group="habitat/task/actions",
    name="discrete_move_forward",
    node=DiscreteMoveForwardActionConfig,
)

cs.store( ##
    package="habitat.task.actions.discrete_turn_left",
    group="habitat/task/actions",
    name="discrete_turn_left",
    node=DiscreteTurnLeftActionConfig,
)
cs.store( ##
    package="habitat.task.actions.discrete_turn_right",
    group="habitat/task/actions",
    name="discrete_turn_right",
    node=DiscreteTurnRightActionConfig,
)
cs.store(
    package="habitat.task.actions.oracle_nav_action",
    group="habitat/task/actions",
    name="oracle_nav_action",
    node=OracleNavActionWOPDDLConfig,
)
cs.store(
    package="habitat.task.actions.oracle_follow_action",
    group="habitat/task/actions",
    name="oracle_follow_action",
    node=OracleFollowActionConfig,
)
