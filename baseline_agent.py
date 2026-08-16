import habitat
import warnings

warnings.filterwarnings("ignore")

import evt_bench  # noqa: F401 — 注册 OracleNavCoordinateActionForRobot 等自定义 action

import copy
import json
import math
import os
import os.path as osp
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np

try:
    import rvo2 as _rvo2

    _RVO2_AVAILABLE = True
except ImportError:
    _RVO2_AVAILABLE = False
from habitat.config.default_structured_configs import AgentConfig
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel
from tqdm import trange

import magnum as mn


# -----------------------------------------------------------------------------
# Base pose / geometry (for logging 与 GT 可见性近似)
# -----------------------------------------------------------------------------


def _as_np3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(3)


def _xz(v: np.ndarray) -> np.ndarray:
    return np.asarray([float(v[0]), float(v[2])], dtype=np.float32)


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _target_pose_from_base_velocity(base_velocity: List[float], dt: float) -> List[float]:
    """Integrate normalized base_velocity for one control step in body frame."""
    vf, vl, wz = (float(base_velocity[0]), float(base_velocity[1]), float(base_velocity[2]))
    return [vf * dt, vl * dt, wz * dt]


def _body_action_from_target_pose(target_pose: List[float]) -> List[float]:
    """NavVLA / LeRobot v3 body-frame waypoint: [dx, dy, dz, dyaw]."""
    return [float(target_pose[0]), float(target_pose[1]), 0.0, float(target_pose[2])]


def _build_step_info_record(
    *,
    frame_index: int,
    ctrl_dt: float,
    iter_step: int,
    robot_pos_before: np.ndarray,
    robot_yaw_before: float,
    robot_pos_after: np.ndarray,
    robot_yaw_after: float,
    human_pos_before: np.ndarray,
    human_yaw_before: float,
    human_pos_after: np.ndarray,
    human_yaw_after: float,
    action: List[float],
    base_velocity_physical: List[float],
    target_pose: List[float],
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-step *_info.json row aligned with rgb_list[frame_index] for LeRobot v3 conversion."""
    timestamp = float(frame_index) * float(ctrl_dt)
    hf = _measure_scalar(info.get("human_following", 0.0))
    hc = _measure_scalar(info.get("human_collision", 0.0))
    col = _measure_scalar(info.get("collision", 0.0))
    body_action = _body_action_from_target_pose(target_pose)
    return {
        "frame_index": int(frame_index),
        "timestamp": timestamp,
        "step": int(iter_step),
        "robot_pos": robot_pos_before.tolist(),
        "robot_yaw": float(robot_yaw_before),
        "human_pos": human_pos_before.tolist(),
        "human_yaw": float(human_yaw_before),
        "robot_pos_after": robot_pos_after.tolist(),
        "robot_yaw_after": float(robot_yaw_after),
        "human_pos_after": human_pos_after.tolist(),
        "human_yaw_after": float(human_yaw_after),
        "base_velocity_physical": base_velocity_physical,
        "action": body_action,
        "target_pose": target_pose,
        "base_velocity": [float(x) for x in action],
        "action_available": True,
        "facing": hf,
        "human_following": hf,
        "dis_to_human": float(np.linalg.norm(robot_pos_after - human_pos_after)),
        "collision": col,
        "human_collision": hc,
    }


def _physical_base_velocity(base_velocity: List[float], action_obj: Any) -> List[float]:
    """Denormalize base_velocity to body-frame m/s and rad/s (BaseVelNonCylinderAction.step)."""
    if action_obj is None:
        return [0.0, 0.0, 0.0]
    vf = float(np.clip(float(base_velocity[0]), -1.0, 1.0)) * float(action_obj._longitudinal_lin_speed)
    if not bool(getattr(action_obj, "_allow_back", True)):
        vf = max(vf, 0.0)
    vl = float(np.clip(float(base_velocity[1]), -1.0, 1.0)) * float(action_obj._lateral_lin_speed)
    wz = float(np.clip(float(base_velocity[2]), -1.0, 1.0)) * float(action_obj._ang_speed)
    return [vf, vl, wz]


def _normalized_base_velocity(physical: List[float], action_obj: Any) -> List[float]:
    """Normalize body-frame m/s and rad/s back to [-1, 1] base_velocity."""
    if action_obj is None:
        return [0.0, 0.0, 0.0]
    lon = float(action_obj._longitudinal_lin_speed)
    lat = float(action_obj._lateral_lin_speed)
    ang = float(action_obj._ang_speed)
    vf = float(np.clip(float(physical[0]) / max(lon, 1e-6), -1.0, 1.0))
    if not bool(getattr(action_obj, "_allow_back", True)):
        vf = max(vf, 0.0)
    vl = float(np.clip(float(physical[1]) / max(lat, 1e-6), -1.0, 1.0))
    wz = float(np.clip(float(physical[2]) / max(ang, 1e-6), -1.0, 1.0))
    return [vf, vl, wz]


def _project_linear_velocity_to_action_limits(
    long_speed: float, lat_speed: float, action_obj: Any
) -> Tuple[float, float]:
    """Preserve direction while fitting a holonomic velocity into axis limits."""
    if action_obj is None:
        return 0.0, 0.0
    lon = max(float(action_obj._longitudinal_lin_speed), 1e-6)
    lat = max(float(action_obj._lateral_lin_speed), 1e-6)
    scale = max(1.0, abs(float(long_speed)) / lon, abs(float(lat_speed)) / lat)
    return float(long_speed) / scale, float(lat_speed) / scale


def _validate_teacher_executor_speed_alignment(
    oracle_action: Any, base_vel_action: Any
) -> None:
    """Fail early if normalized teacher commands use different physical caps."""
    if oracle_action is None or base_vel_action is None:
        return
    fields = (
        ("longitudinal", "_longitudinal_lin_speed"),
        ("lateral", "_lateral_lin_speed"),
        ("yaw", "_ang_speed"),
    )
    mismatches = []
    for label, attr in fields:
        teacher_value = float(getattr(oracle_action, attr))
        executor_value = float(getattr(base_vel_action, attr))
        if not np.isclose(teacher_value, executor_value, rtol=0.0, atol=1e-6):
            mismatches.append(
                f"{label}: teacher={teacher_value:g}, executor={executor_value:g}"
            )
    if mismatches:
        raise ValueError(
            "Oracle teacher/base-velocity speed caps must match because both share "
            "the same normalized action labels. " + "; ".join(mismatches)
        )


def _right_xz_unit(fwd: np.ndarray) -> np.ndarray:
    return np.asarray([float(fwd[1]), -float(fwd[0])], dtype=np.float32)


def _base_rotation3(robot_agent: Any) -> np.ndarray:
    # BaseVelNonCylinderAction integrates local velocity against
    # ``sim_obj.transformation``. Spot/Stretch/Fetch ``base_transformation``
    # adds a fixed -90° X rotation for camera/body conventions, so using it
    # here maps planar lateral velocity onto the vertical axis and turns an
    # intended radial retreat into saturated lateral motion.
    if hasattr(robot_agent, "sim_obj") and hasattr(
        robot_agent.sim_obj, "transformation"
    ):
        T = robot_agent.sim_obj.transformation
    else:
        T = _get_base_matrix4(robot_agent)
    m = np.array(T, dtype=np.float64).reshape(4, 4)
    return m[:3, :3]


def _basevel_local_lin_to_world_xz(
    long_speed: float, lat_speed: float, robot_agent: Any
) -> np.ndarray:
    """与 BaseVelNonCylinderAction 一致：local (long, 0, -lat) -> world XZ。"""
    R = _base_rotation3(robot_agent)
    v_world = R[:, 0] * float(long_speed) - R[:, 2] * float(lat_speed)
    return np.asarray([float(v_world[0]), float(v_world[2])], dtype=np.float32)


def _basevel_world_xz_to_local_lin(
    vec_world: np.ndarray, robot_agent: Any
) -> Tuple[float, float]:
    """world XZ -> local long (+x) 与 lateral（对应 local -z 速度分量）。"""
    R = _base_rotation3(robot_agent)
    A = np.asarray(
        [
            [float(R[0, 0]), float(-R[0, 2])],
            [float(R[2, 0]), float(-R[2, 2])],
        ],
        dtype=np.float64,
    )
    b = np.asarray([float(vec_world[0]), float(vec_world[1])], dtype=np.float64)
    try:
        long_speed, lat_speed = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        long_speed, lat_speed, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(long_speed), float(lat_speed)


def _lateral_speed_affects_xz(robot_agent: Any, eps: float = 0.15) -> bool:
    """local -z 速度在 XZ 上是否有足够分量（Spot 上常为 False）。"""
    R = _base_rotation3(robot_agent)
    lat_xz = np.asarray([-float(R[0, 2]), float(-R[2, 2])], dtype=np.float64)
    return float(np.linalg.norm(lat_xz)) > eps


def _clip_world_vel(vel: np.ndarray, max_speed: float) -> np.ndarray:
    speed = float(np.linalg.norm(vel))
    if speed <= max_speed or speed < 1e-6:
        return vel
    return vel * (max_speed / speed)


def _get_agent_xz(agent: Any) -> np.ndarray:
    p = _get_base_pos(agent)
    return np.asarray([float(p[0]), float(p[2])], dtype=np.float32)


def _humanoid_agent_indices(sim) -> List[int]:
    """All humanoids (incl. main target agent_0), excluding the robot (agent_1)."""
    return [i for i in range(len(sim.agents_mgr)) if i != 1]


def _avoid_other_pedestrians_enabled() -> bool:
    """True: RVO 对所有 humanoid 避障；False: 仅对主行人 agent_0 避障。"""
    return os.environ.get("AVOID_OTHER_PEDESTRIANS", "1") != "0"


def _humanoids_for_rvo_avoidance(sim) -> List[int]:
    """参与 RVO 动态避障的 humanoid 索引。"""
    indices = _humanoid_agent_indices(sim)
    if _avoid_other_pedestrians_enabled():
        return indices
    return [idx for idx in indices if idx == 0]


def _collect_agent_xz_by_index(sim, indices: List[int]) -> Dict[int, np.ndarray]:
    return {
        idx: _get_agent_xz(sim.agents_mgr[idx].articulated_agent)
        for idx in indices
    }


def _estimate_xz_velocity(
    pos_now: np.ndarray,
    pos_prev: Optional[np.ndarray],
    dt: float,
    previous_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    if pos_prev is None:
        return np.zeros(2, dtype=np.float32)
    raw_velocity = (pos_now - pos_prev) / max(float(dt), 1e-6)
    if previous_velocity is None:
        return np.asarray(raw_velocity, dtype=np.float32)
    try:
        alpha_value = float(os.environ.get("RVO_VELOCITY_EMA_ALPHA", "0.35"))
    except ValueError:
        alpha_value = 0.35
    alpha = float(np.clip(alpha_value, 0.0, 1.0))
    return np.asarray(
        (1.0 - alpha) * previous_velocity + alpha * raw_velocity,
        dtype=np.float32,
    )


def _rvo_radius_for_agent(agent_idx: int, base_radius: float) -> float:
    """Use a full safety radius for the main target unless explicitly reduced."""
    if agent_idx == 0:
        scale = float(os.environ.get("LEADER_RVO_RADIUS_SCALE", "1.0"))
    else:
        scale = float(os.environ.get("OTHER_RVO_RADIUS_SCALE", "1.0"))
    return max(0.05, base_radius * scale)


def _correct_action_with_rvo2(
    raw_action: List[float],
    robot_agent: Any,
    sim,
    prev_xz_by_agent: Optional[Dict[int, np.ndarray]],
    ctrl_dt: float,
    base_vel_action: Any,
    velocity_ema_by_agent: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[List[float], float]:
    """ORCA via RVO2: raw body vel -> world pref -> safe world vel -> body vel."""
    if not _RVO2_AVAILABLE:
        raise ImportError(
            "DYN_OBSTACLE_AVOID=1 but pyrvo2 is not installed. "
            "See https://github.com/sybrenstuvel/Python-RVO2"
        )

    dt = max(float(ctrl_dt), 1e-3)
    neighbor_dist = float(os.environ.get("RVO_NEIGHBOR_DIST", "2.5"))
    max_neighbors = int(os.environ.get("RVO_MAX_NEIGHBORS", "10"))
    time_horizon = float(os.environ.get("RVO_TIME_HORIZON", "2.0"))
    time_horizon_obst = float(os.environ.get("RVO_TIME_HORIZON_OBST", "2.0"))
    agent_radius = float(os.environ.get("RVO_AGENT_RADIUS", "0.35"))
    executor_max_speed = max(
        float(base_vel_action._longitudinal_lin_speed),
        float(base_vel_action._lateral_lin_speed),
    )
    max_speed = float(os.environ.get("RVO_MAX_SPEED", str(executor_max_speed)))

    robot_idx = 1
    robot_xz = _get_agent_xz(robot_agent)
    physical = _physical_base_velocity(raw_action, base_vel_action)
    pref_world = _clip_world_vel(
        _basevel_local_lin_to_world_xz(physical[0], physical[1], robot_agent),
        max_speed,
    )

    robot_prev = prev_xz_by_agent.get(robot_idx) if prev_xz_by_agent else None
    robot_prev_velocity = (
        velocity_ema_by_agent.get(robot_idx)
        if velocity_ema_by_agent is not None
        else None
    )
    robot_vel_world = _estimate_xz_velocity(
        robot_xz, robot_prev, dt, robot_prev_velocity
    )
    if velocity_ema_by_agent is not None:
        velocity_ema_by_agent[robot_idx] = robot_vel_world.copy()

    rvo_sim = _rvo2.PyRVOSimulator(
        dt,
        neighbor_dist,
        max_neighbors,
        time_horizon,
        time_horizon_obst,
        agent_radius,
        max_speed,
    )

    robot_id = rvo_sim.addAgent((float(robot_xz[0]), float(robot_xz[1])))
    rvo_sim.setAgentVelocity(
        robot_id,
        (float(robot_vel_world[0]), float(robot_vel_world[1])),
    )
    rvo_sim.setAgentPrefVelocity(
        robot_id,
        (float(pref_world[0]), float(pref_world[1])),
    )

    for human_idx in _humanoids_for_rvo_avoidance(sim):
        human_xz = _get_agent_xz(sim.agents_mgr[human_idx].articulated_agent)
        human_prev = prev_xz_by_agent.get(human_idx) if prev_xz_by_agent else None
        human_prev_velocity = (
            velocity_ema_by_agent.get(human_idx)
            if velocity_ema_by_agent is not None
            else None
        )
        human_vel = _estimate_xz_velocity(
            human_xz, human_prev, dt, human_prev_velocity
        )
        if velocity_ema_by_agent is not None:
            velocity_ema_by_agent[human_idx] = human_vel.copy()

        human_id = rvo_sim.addAgent((float(human_xz[0]), float(human_xz[1])))
        rvo_sim.setAgentRadius(human_id, _rvo_radius_for_agent(human_idx, agent_radius))
        rvo_sim.setAgentVelocity(
            human_id,
            (float(human_vel[0]), float(human_vel[1])),
        )
        rvo_sim.setAgentPrefVelocity(
            human_id,
            (float(human_vel[0]), float(human_vel[1])),
        )

    rvo_sim.doStep()
    safe_world = np.array(rvo_sim.getAgentVelocity(robot_id), dtype=np.float32)
    safe_long, safe_lat = _basevel_world_xz_to_local_lin(safe_world, robot_agent)
    # RVO 只在 XZ 平面；Spot 等机器人 lateral(-local.z) 可能不在 XZ 上，需保留 teacher。
    if not _lateral_speed_affects_xz(robot_agent):
        safe_lat = float(physical[1])
    correction_mag = float(np.linalg.norm(safe_world - pref_world))

    safe_long, safe_lat = _project_linear_velocity_to_action_limits(
        safe_long, safe_lat, base_vel_action
    )

    safe_physical = [safe_long, safe_lat, float(physical[2])]
    safe_action = _normalized_base_velocity(safe_physical, base_vel_action)
    return safe_action, correction_mag


def _apply_dynamic_avoidance(
    raw_action: List[float],
    robot_agent: Any,
    sim,
    prev_xz_by_agent: Optional[Dict[int, np.ndarray]],
    ctrl_dt: float,
    base_vel_action: Any,
    velocity_ema_by_agent: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[List[float], float]:
    if os.environ.get("DYN_OBSTACLE_AVOID", "1") == "0":
        return raw_action, 0.0
    return _correct_action_with_rvo2(
        raw_action,
        robot_agent,
        sim,
        prev_xz_by_agent,
        ctrl_dt,
        base_vel_action,
        velocity_ema_by_agent,
    )


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return float(default)


def _main_human_guard_runtime_config() -> Dict[str, Any]:
    return {
        "version": "predictive_relative_v3_velocity_frame",
        "enabled": os.environ.get("MAIN_HUMAN_PROXIMITY_GUARD", "1") != "0",
        "start_distance": _env_float("MAIN_HUMAN_GUARD_START_DISTANCE", 1.8),
        "stop_distance": _env_float("MAIN_HUMAN_GUARD_STOP_DISTANCE", 1.0),
        "approach_gain": _env_float("MAIN_HUMAN_GUARD_APPROACH_GAIN", 1.5),
        "closing_margin": _env_float("MAIN_HUMAN_GUARD_CLOSING_MARGIN", 0.25),
        "prediction_horizon": _env_float(
            "MAIN_HUMAN_GUARD_PREDICTION_HORIZON", 0.25
        ),
        "emergency_distance": _env_float(
            "MAIN_HUMAN_GUARD_EMERGENCY_DISTANCE", 1.15
        ),
        "min_retreat_speed": _env_float(
            "MAIN_HUMAN_GUARD_MIN_RETREAT_SPEED", 0.8
        ),
        "max_tangent_speed": _env_float(
            "MAIN_HUMAN_GUARD_MAX_TANGENT_SPEED", 0.15
        ),
        "max_yaw_speed": _env_float("MAIN_HUMAN_GUARD_MAX_YAW_SPEED", 0.5),
        "max_human_speed": _env_float(
            "MAIN_HUMAN_GUARD_MAX_HUMAN_SPEED", 2.5
        ),
        "leader_rvo_radius_scale": _env_float("LEADER_RVO_RADIUS_SCALE", 1.0),
    }


def _apply_main_human_proximity_guard(
    action: List[float],
    robot_agent: Any,
    human_agent: Any,
    base_vel_action: Any,
    human_velocity_world: Optional[np.ndarray] = None,
    ctrl_dt: float = 0.1,
) -> Tuple[List[float], float, Dict[str, Any]]:
    """Apply a predictive relative-velocity guard after all other corrections.

    The main human moves first inside the composite Habitat step, so a guard
    based only on the robot's absolute approach speed can be stale by the time
    the robot action executes.  This projection includes the observed human
    radial velocity, predicts near-term clearance, forces radial retreat in an
    emergency, and suppresses tangent/yaw motion that can arc back into the
    human at very short range.
    """
    debug: Dict[str, Any] = {
        "distance": -1.0,
        "predicted_distance": -1.0,
        "human_radial_speed": 0.0,
        "toward_speed": 0.0,
        "toward_speed_limit": -1.0,
        "toward_speed_after": 0.0,
        "emergency": False,
    }
    guard_config = _main_human_guard_runtime_config()
    if (
        not guard_config["enabled"]
        or base_vel_action is None
    ):
        return action, 0.0, debug

    robot_xz = _get_agent_xz(robot_agent)
    human_xz = _get_agent_xz(human_agent)
    rel_human = human_xz - robot_xz
    distance = float(np.linalg.norm(rel_human))
    debug["distance"] = distance
    if distance < 1e-6:
        toward_human = np.zeros(2, dtype=np.float32)
    else:
        toward_human = rel_human / distance

    human_velocity = np.asarray(
        human_velocity_world
        if human_velocity_world is not None
        else np.zeros(2, dtype=np.float32),
        dtype=np.float32,
    ).reshape(2)
    if not np.all(np.isfinite(human_velocity)):
        human_velocity = np.zeros(2, dtype=np.float32)
    human_velocity = _clip_world_vel(
        human_velocity,
        max(0.1, float(guard_config["max_human_speed"])),
    )
    human_radial_speed = (
        float(np.dot(human_velocity, toward_human)) if distance >= 1e-6 else 0.0
    )

    stop_distance = max(
        0.05, float(guard_config["stop_distance"])
    )
    start_distance = max(
        stop_distance + 1e-3,
        float(guard_config["start_distance"]),
    )
    approach_gain = max(
        0.0, float(guard_config["approach_gain"])
    )
    closing_margin = max(
        0.0, float(guard_config["closing_margin"])
    )
    prediction_horizon = max(
        0.0, float(guard_config["prediction_horizon"])
    )
    predicted_distance = max(
        0.0, distance + human_radial_speed * prediction_horizon
    )
    debug["predicted_distance"] = predicted_distance
    debug["human_radial_speed"] = human_radial_speed

    if distance >= start_distance and predicted_distance >= start_distance:
        return action, 0.0, debug

    physical = _physical_base_velocity(action, base_vel_action)
    world_velocity = _basevel_local_lin_to_world_xz(
        physical[0], physical[1], robot_agent
    )
    if distance < 1e-6:
        safe_world = np.zeros(2, dtype=np.float32)
        toward_speed = float(np.linalg.norm(world_velocity))
        max_toward_speed = 0.0
        safe_yaw_speed = 0.0
        emergency = True
    else:
        toward_speed = float(np.dot(world_velocity, toward_human))
        max_toward_speed = (
            human_radial_speed
            + approach_gain * max(distance - stop_distance, 0.0)
            - closing_margin
        )
        emergency_distance = max(
            stop_distance,
            float(guard_config["emergency_distance"]),
        )
        emergency = (
            distance < emergency_distance
            or predicted_distance < emergency_distance
        )

        tangent_velocity = world_velocity - toward_speed * toward_human
        if emergency:
            min_retreat_speed = max(
                0.0,
                float(guard_config["min_retreat_speed"]),
            )
            required_retreat_speed = max(
                0.0,
                (stop_distance - predicted_distance)
                / max(float(ctrl_dt), 1e-3),
            )
            max_toward_speed = min(
                max_toward_speed,
                -min_retreat_speed,
                -required_retreat_speed,
            )
            max_tangent_speed = max(
                0.0,
                float(guard_config["max_tangent_speed"]),
            )
            tangent_velocity = _clip_world_vel(
                tangent_velocity, max_tangent_speed
            )
            max_emergency_yaw = max(
                0.0,
                float(guard_config["max_yaw_speed"]),
            )
            safe_yaw_speed = float(
                np.clip(physical[2], -max_emergency_yaw, max_emergency_yaw)
            )
        else:
            safe_yaw_speed = float(physical[2])

        safe_toward_speed = min(toward_speed, max_toward_speed)
        safe_world = safe_toward_speed * toward_human + tangent_velocity

    debug["toward_speed"] = toward_speed
    debug["toward_speed_limit"] = max_toward_speed
    debug["emergency"] = emergency

    safe_long, safe_lat = _basevel_world_xz_to_local_lin(
        safe_world, robot_agent
    )
    safe_long, safe_lat = _project_linear_velocity_to_action_limits(
        safe_long, safe_lat, base_vel_action
    )
    safe_action = _normalized_base_velocity(
        [safe_long, safe_lat, safe_yaw_speed], base_vel_action
    )
    projected_safe_world = _basevel_local_lin_to_world_xz(
        safe_long, safe_lat, robot_agent
    )
    debug["toward_speed_after"] = (
        float(np.dot(projected_safe_world, toward_human))
        if distance >= 1e-6
        else 0.0
    )
    correction_mag = float(
        np.linalg.norm(projected_safe_world - world_velocity)
    )
    yaw_correction = abs(safe_yaw_speed - float(physical[2]))
    if correction_mag < 1e-8 and yaw_correction < 1e-8:
        return action, 0.0, debug
    return safe_action, correction_mag, debug


def _yaw_from_quat_like(q: Any) -> float:
    try:
        x, y, z = float(q.vector.x), float(q.vector.y), float(q.vector.z)
        w = float(q.scalar)
        siny_cosp = 2.0 * (w * y + x * z)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return _wrap_pi(math.atan2(siny_cosp, cosy_cosp))
    except Exception:
        return 0.0


def _get_base_pos(agent: Any) -> np.ndarray:
    if hasattr(agent, "base_pos"):
        return _as_np3(agent.base_pos)
    if hasattr(agent, "base_transformation"):
        t = agent.base_transformation
        try:
            return _as_np3(t.translation)
        except Exception:
            pass
    raise AttributeError("Cannot read articulated agent base position")


def _get_base_yaw(agent: Any) -> float:
    if hasattr(agent, "base_rot"):
        r = agent.base_rot
        try:
            return _wrap_pi(float(r))
        except Exception:
            pass
        return _yaw_from_quat_like(r)
    if hasattr(agent, "base_transformation"):
        try:
            return _yaw_from_quat_like(agent.base_transformation.rotation())
        except Exception:
            pass
    return 0.0


def _get_base_matrix4(agent: Any) -> mn.Matrix4:
    if hasattr(agent, "base_transformation"):
        return agent.base_transformation
    raise AttributeError("articulated_agent.base_transformation missing")


def _forward_xz_unit_from_base(agent: Any) -> np.ndarray:
    T = _get_base_matrix4(agent)
    p0 = T.transform_point(mn.Vector3(0.0, 0.0, 0.0))
    p1 = T.transform_point(mn.Vector3(0.0, 0.0, -1.0))
    fx = float(p1.x - p0.x)
    fz = float(p1.z - p0.z)
    n = math.hypot(fx, fz) + 1e-9
    return np.asarray([fx / n, fz / n], dtype=np.float32)


def _yaw_error_to_world_point_xz(robot_agent: Any, robot_pos: np.ndarray, target_world: np.ndarray) -> float:
    fwd = _forward_xz_unit_from_base(robot_agent)
    dx = float(target_world[0] - robot_pos[0])
    dz = float(target_world[2] - robot_pos[2])
    target_yaw = math.atan2(dx, dz)
    robot_yaw = math.atan2(float(fwd[0]), float(fwd[1]))
    return _wrap_pi(target_yaw - robot_yaw)


def _visible_human_in_robot_fov_gt(
    robot_agent: Any,
    robot_pos: np.ndarray,
    human_pos: np.ndarray,
    max_dist: float = 10.0,
    half_fov_rad: float = 1.15,
) -> bool:
    """仿真真值近似：人在水平距离内且落在机身前向半角锥内。"""
    d = float(np.linalg.norm(_xz(human_pos - robot_pos)))
    if d > max_dist or d < 1e-4:
        return False
    yaw_err = abs(_yaw_error_to_world_point_xz(robot_agent, robot_pos, human_pos))
    return yaw_err <= half_fov_rad


class GTBBoxAgent(AgentConfig):
    """记录 RGB、debug 与 episode 视频；机器人控制仅通过 oracle_robot_teacher（见 evaluate_agent）。"""

    VIDEO_VIEW_SENSOR_KEYS = {
        "front": "agent_1_articulated_agent_jaw_rgb",
        "left": "agent_1_articulated_agent_left_rgb",
        "right": "agent_1_articulated_agent_right_rgb",
        "rear": "agent_1_articulated_agent_back_rgb",
    }

    def __init__(self, result_path: str, target_id: Optional[int] = None):
        super().__init__()
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.target_id = target_id
        self.rgb_list: List[np.ndarray] = []
        self.rgb_lists: Dict[str, List[np.ndarray]] = {
            view: [] for view in self.VIDEO_VIEW_SENSOR_KEYS
        }
        self.debug: Dict[str, Any] = {}
        self.save_fps: int = 10

    def append_video_frames(self, observations: Dict[str, Any]) -> None:
        for view, sensor_key in self.VIDEO_VIEW_SENSOR_KEYS.items():
            if sensor_key not in observations:
                continue
            rgb = observations[sensor_key][:, :, :3]
            frame = np.asarray(rgb, dtype=np.uint8).copy()
            self.rgb_lists.setdefault(view, []).append(frame)
            if view == "front":
                self.rgb_list.append(frame)

    def video_frames_snapshot(self, n_frames: Optional[int] = None) -> Dict[str, List[np.ndarray]]:
        out: Dict[str, List[np.ndarray]] = {}
        for view, frames in self.rgb_lists.items():
            selected = frames if n_frames is None else frames[:n_frames]
            if selected:
                out[view] = [np.array(f, copy=True) for f in selected]
        if "front" not in out and self.rgb_list:
            selected = self.rgb_list if n_frames is None else self.rgb_list[:n_frames]
            out["front"] = [np.array(f, copy=True) for f in selected]
        return out

    def set_video_frames(self, frames_by_view: Dict[str, List[np.ndarray]]) -> None:
        self.rgb_lists = {
            view: [np.array(f, copy=True) for f in frames_by_view.get(view, [])]
            for view in self.VIDEO_VIEW_SENSOR_KEYS
        }
        self.rgb_list = [np.array(f, copy=True) for f in self.rgb_lists.get("front", [])]

    def reset(self, episode: NavigationEpisode = None, success: bool = False):
        has_frames = any(len(frames) != 0 for frames in self.rgb_lists.values()) or len(self.rgb_list) != 0
        if has_frames and episode is not None and success:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split(".")[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            frames_by_view = self.video_frames_snapshot()
            for view in ("front", "left", "right", "rear"):
                frames = frames_by_view.get(view, [])
                if not frames:
                    continue
                output_video_path = os.path.join(save_dir, f"{episode.episode_id}_{view}.mp4")
                imageio.mimsave(output_video_path, frames, fps=int(self.save_fps))
            saved_views = ",".join(view for view, frames in frames_by_view.items() if frames)
            print(
                f"Successfully saved episode videos with episode id {episode.episode_id} "
                f"views=[{saved_views}]"
            )

        self.rgb_list = []
        self.rgb_lists = {view: [] for view in self.VIDEO_VIEW_SENSOR_KEYS}
        self.debug = {}

    def _target_visible(self, observations: Dict[str, Any], detector: Dict[str, Any]) -> bool:
        try:
            d = detector.get("agent_1_main_humanoid_detector_sensor", {})
            if bool(d.get("facing", False)):
                return True
        except Exception:
            pass

        if self.target_id is not None and "agent_1_articulated_agent_jaw_panoptic" in observations:
            panoptic = observations["agent_1_articulated_agent_jaw_panoptic"]
            mask = panoptic == self.target_id
            if hasattr(mask, "ndim") and mask.ndim == 3:
                mask = mask[:, :, 0]
            return bool(np.any(mask))
        return False


_TRACK_COMPOSITE_ACTION_NAMES = (
    "agent_0_humanoid_navigate_action",
    "agent_1_base_velocity",
    "agent_2_oracle_nav_randcoord_action_obstacle",
    "agent_3_oracle_nav_randcoord_action_obstacle",
    "agent_4_oracle_nav_randcoord_action_obstacle",
    "agent_5_oracle_nav_randcoord_action_obstacle",
)


def _measure_scalar(x: Any, default: float = 0.0) -> float:
    """habitat 度量可能是 float / numpy 标量 / 0 维数组，统一成 Python float。"""
    if x is None:
        return default
    try:
        a = np.asarray(x).reshape(-1)
        if a.size == 0:
            return default
        return float(a[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return default


def _rollout_failure_reasons(
    result: Dict[str, Any],
    record_infos: List[Dict[str, Any]],
    status: str,
    min_following_rate: float = 0.5,
) -> List[str]:
    """Return stable reason codes for an episode that failed rollout criteria."""
    reasons: List[str] = []
    status_reason = {
        "Lost": "lost_main_human",
        "HumanStuck": "main_human_stuck",
        "Collision": "human_collision",
    }.get(status)
    if status_reason is not None:
        reasons.append(status_reason)
    elif status != "Normal":
        reasons.append(f"status_{status.lower()}")

    if len(record_infos) == 0:
        reasons.append("empty_trajectory")
    if not bool(result.get("finish", False)):
        reasons.append("episode_incomplete")
    if (
        _measure_scalar(result.get("collision", 0.0)) >= 1.0 - 1e-6
        and "human_collision" not in reasons
    ):
        reasons.append("human_collision")
    if not (
        bool(result.get("human_following_success", False))
        or bool(result.get("last_human_following", False))
    ):
        reasons.append("following_not_achieved")
    if not bool(result.get("success_visible_detector", False)):
        reasons.append("target_not_visible_at_end")
    if float(result.get("following_rate", 0.0)) < float(min_following_rate):
        reasons.append("low_following_rate")

    deduplicated: List[str] = []
    for reason in reasons:
        if reason not in deduplicated:
            deduplicated.append(reason)
    return deduplicated or ["rollout_rejected_unknown"]


_ROLLOUT_FAILURE_MESSAGES = {
    "lost_main_human": "连续多步与主行人距离过远，判定跟丢",
    "main_human_stuck": "主行人连续多步未移动，episode 被丢弃",
    "human_collision": "检测到机器人与主行人碰撞",
    "empty_trajectory": "episode 未产生可保存的轨迹 step",
    "episode_incomplete": "episode 在环境自然结束前提前终止",
    "following_not_achieved": "未达到主行人跟随成功条件",
    "target_not_visible_at_end": "episode 结束时主行人不在目标检测视野内",
    "low_following_rate": "有效跟随 step 比例低于 0.5",
    "rollout_rejected_unknown": "未满足 rollout 保存条件，原因未分类",
}


def _append_rollout_failure_log(
    save_path: str,
    split_id: Optional[int],
    scene_key: str,
    episode_id: str,
    result: Dict[str, Any],
    record_infos: List[Dict[str, Any]],
    reasons: List[str],
) -> str:
    """Append one JSONL record without letting logging abort the rollout."""
    log_dir = os.path.join(save_path, "runtime_logs")
    split_label = str(split_id) if split_id is not None else f"pid_{os.getpid()}"
    log_path = os.path.join(log_dir, f"split_{split_label}_episode_failures.jsonl")

    distances = [
        _measure_scalar(row.get("dis_to_human"))
        for row in record_infos
        if row.get("dis_to_human") is not None
        and np.isfinite(_measure_scalar(row.get("dis_to_human")))
    ]
    guard_corrections = [
        _measure_scalar(row.get("main_human_guard_correction", 0.0))
        for row in record_infos
    ]
    last_record = record_infos[-1] if record_infos else {}
    tail_steps = max(1, int(_env_float("ROLLOUT_FAILURE_TAIL_STEPS", 20)))
    recent_steps = []
    for row in record_infos[-tail_steps:]:
        recent_steps.append(
            {
                "step": int(row.get("step", 0)),
                "distance_to_human": _measure_scalar(
                    row.get("dis_to_human"), -1.0
                ),
                "robot_pos": row.get("robot_pos"),
                "robot_pos_after": row.get("robot_pos_after"),
                "human_pos": row.get("human_pos"),
                "human_pos_after": row.get("human_pos_after"),
                "base_velocity_physical": row.get("base_velocity_physical"),
                "base_velocity_raw": row.get("base_velocity_raw"),
                "guard_correction": _measure_scalar(
                    row.get("main_human_guard_correction", 0.0)
                ),
                "guard_predicted_distance": _measure_scalar(
                    row.get("main_human_guard_predicted_distance"), -1.0
                ),
                "guard_human_radial_speed": _measure_scalar(
                    row.get("main_human_guard_human_radial_speed", 0.0)
                ),
                "guard_toward_speed": _measure_scalar(
                    row.get("main_human_guard_toward_speed", 0.0)
                ),
                "guard_toward_speed_limit": _measure_scalar(
                    row.get("main_human_guard_toward_speed_limit", -1.0)
                ),
                "guard_toward_speed_after": _measure_scalar(
                    row.get("main_human_guard_toward_speed_after", 0.0)
                ),
                "guard_emergency": bool(
                    row.get("main_human_guard_emergency", False)
                ),
                "human_collision": bool(
                    _measure_scalar(row.get("human_collision", 0.0))
                    >= 1.0 - 1e-6
                ),
            }
        )
    payload = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "pid": int(os.getpid()),
        "split_id": split_id,
        "scene_id": scene_key,
        "episode_id": episode_id,
        "status": str(result.get("status", "Unknown")),
        "primary_reason": reasons[0],
        "primary_reason_message": _ROLLOUT_FAILURE_MESSAGES.get(
            reasons[0], reasons[0]
        ),
        "failure_reasons": reasons,
        "failure_reason_messages": [
            _ROLLOUT_FAILURE_MESSAGES.get(reason, reason) for reason in reasons
        ],
        "finish": bool(result.get("finish", False)),
        "steps": int(result.get("total_step", len(record_infos))),
        "following_steps": int(result.get("following_step", 0)),
        "following_rate": float(result.get("following_rate", 0.0)),
        "human_following_success": bool(
            result.get("human_following_success", False)
        ),
        "last_human_following": bool(result.get("last_human_following", False)),
        "target_visible_at_end": bool(
            result.get("success_visible_detector", False)
        ),
        "human_collision": bool(
            _measure_scalar(result.get("collision", 0.0)) >= 1.0 - 1e-6
        ),
        "min_distance_to_human": min(distances) if distances else None,
        "final_distance_to_human": distances[-1] if distances else None,
        "max_main_human_guard_correction": (
            max(guard_corrections) if guard_corrections else 0.0
        ),
        "main_human_guard_config": result.get("main_human_guard_config"),
        "last_base_velocity_physical": last_record.get(
            "base_velocity_physical"
        ),
        "recent_steps": recent_steps,
    }
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as failure_log:
            failure_log.write(
                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            )
    except Exception as exc:
        print(
            f"[WARN] failed to append rollout failure log for "
            f"{scene_key}/{episode_id}: {exc}"
        )
    return log_path


def _human_stuck_filter_enabled() -> bool:
    return os.environ.get("HUMAN_STUCK_FILTER", "1") != "0"


def _human_stuck_xz_eps() -> float:
    """主行人单步 xz 位移低于此阈值视为「未移动」（L1 距离，单位 m）。"""
    try:
        return float(os.environ.get("HUMAN_STUCK_XZ_EPS", "1e-4"))
    except ValueError:
        return 1e-4


def _human_stuck_min_steps() -> int:
    """连续未移动步数达到此值则判定 stuck 并丢弃 episode。"""
    try:
        return max(1, int(os.environ.get("HUMAN_STUCK_MIN_STEPS", "60")))
    except ValueError:
        return 30


def _human_xz_step_displacement(
    human_pos_before: np.ndarray, human_pos_after: np.ndarray
) -> float:
    b = _as_np3(human_pos_before)
    a = _as_np3(human_pos_after)
    return float(abs(float(a[0]) - float(b[0])) + abs(float(a[2]) - float(b[2])))


def _max_consecutive_human_stuck_steps(
    record_infos: List[Dict[str, Any]], xz_eps: float
) -> int:
    max_run = 0
    run = 0
    for row in record_infos:
        if _human_xz_step_displacement(row["human_pos"], row["human_pos_after"]) < xz_eps:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _agent1_base_velocity_config(config) -> Dict[str, float]:
    """与 track_infer_stt.yaml 中 agent_1_base_velocity 一致，用于写入 result 元数据。"""
    out: Dict[str, float] = {}
    try:
        h = getattr(config, "habitat", config)
        acts = h.task.actions.agent_1_base_velocity
        out["longitudinal_lin_speed"] = float(acts.longitudinal_lin_speed)
        out["lateral_lin_speed"] = float(acts.lateral_lin_speed)
        out["ang_speed"] = float(acts.ang_speed)
    except Exception:
        pass
    return out


def _episode_output_paths(save_path: str, scene_key: str, episode_id: str) -> Dict[str, str]:
    save_dir = os.path.join(save_path, scene_key)
    eid = str(episode_id)
    return {
        "result": os.path.join(save_dir, f"{eid}.json"),
        "info": os.path.join(save_dir, f"{eid}_info.json"),
        "video_front": os.path.join(save_dir, f"{eid}_front.mp4"),
        "video_left": os.path.join(save_dir, f"{eid}_left.mp4"),
        "video_right": os.path.join(save_dir, f"{eid}_right.mp4"),
        "video_rear": os.path.join(save_dir, f"{eid}_rear.mp4"),
    }


def _episode_already_saved(
    save_path: str,
    scene_key: str,
    episode_id: str,
    require_video: Optional[bool] = None,
) -> bool:
    """Resume 时跳过已成功落盘的 episode（json + info，默认还要求 mp4）。"""
    if require_video is None:
        require_video = os.environ.get("SAVE_VIDEO", "1") != "0"
    paths = _episode_output_paths(save_path, scene_key, episode_id)
    if not (os.path.isfile(paths["result"]) and os.path.isfile(paths["info"])):
        return False
    if require_video:
        required_video_keys = ("video_front", "video_left", "video_right", "video_rear")
        if not all(os.path.isfile(paths[key]) for key in required_video_keys):
            return False
    return True


def evaluate_agent(
    config,
    dataset_split,
    save_path,
    target_id=None,
    split_id: Optional[int] = None,
) -> None:
    robot_config = GTBBoxAgent(save_path, target_id)
    base_vel_meta = _agent1_base_velocity_config(config)
    guard_runtime_config = _main_human_guard_runtime_config()
    print(
        "[main-human-guard] "
        + json.dumps(guard_runtime_config, ensure_ascii=False, sort_keys=True)
    )
    debug_save_all = os.environ.get("DATA_DEBUG", "0") == "1"
    if debug_save_all and os.environ.get("DATA_QUIET") != "1":
        print("DATA_DEBUG=1: saving all episodes (including failures, full trajectories)")

    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        robot_config.reset()
        num_episodes = len(env.episodes)
        skipped_episodes = 0
        discarded_stuck_episodes = 0
        failed_episodes = 0
        failure_reason_counts: Dict[str, int] = {}
        failure_log_path: Optional[str] = None
        human_stuck_filter = _human_stuck_filter_enabled()
        human_stuck_eps = _human_stuck_xz_eps()
        human_stuck_min_steps = _human_stuck_min_steps()

        for _ in trange(num_episodes):
            obs = env.reset()

            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split(".")[0]
            episode_id = str(env.current_episode.episode_id)
            if _episode_already_saved(save_path, scene_key, episode_id):
                skipped_episodes += 1
                if os.environ.get("DATA_QUIET") != "1":
                    print(f"[skip] already saved: {scene_key}/{episode_id}")
                continue

            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
            ]
            sim.set_light_setup(light_setup)

            result: Dict[str, Any] = {}
            record_infos: List[Dict[str, Any]] = []

            try:
                instruction = env.current_episode.info.get("instruction", None)
            except Exception:
                instruction = None

            finished = False
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            too_far_count = 0
            human_stuck_count = 0
            status = "Normal"
            ctrl_dt = 1.0 / float(sim.ctrl_freq)
            robot_config.save_fps = int(sim.ctrl_freq)
            prev_xz_by_agent: Optional[Dict[int, np.ndarray]] = None
            velocity_ema_by_agent: Dict[int, np.ndarray] = {}
            base_vel_action = env.task.actions.get("agent_1_base_velocity")
            oracle_follow_action = env.task.actions.get("agent_1_oracle_follow_action")
            _validate_teacher_executor_speed_alignment(
                oracle_follow_action, base_vel_action
            )

            while not env.episode_over:
                # record_info: Dict[str, Any] = {}
                obs = sim.get_sensor_observations()
                detector = env.task._get_observations(env.current_episode)

                robot_pos_before = _get_base_pos(robot_agent)
                human_pos_before = _get_base_pos(humanoid_agent_main)
                robot_yaw_before = _get_base_yaw(robot_agent)
                human_yaw_before = _get_base_yaw(humanoid_agent_main)

                robot_config.append_video_frames(obs)
                visible_det = robot_config._target_visible(obs, detector)
                visible_geom = _visible_human_in_robot_fov_gt(
                    robot_agent, robot_pos_before, human_pos_before
                )
                robot_config.debug = {
                    "mode": "oracle_robot_teacher",
                    "fallback": False,
                    "visible": bool(visible_det or visible_geom),
                    "visible_detector": bool(visible_det),
                    "visible_geom_gt": bool(visible_geom),
                    "expert_goal": None,
                    "expert_next_wp": None,
                    "path_geodesic": None,
                    "snap_dist": None,
                }

                if oracle_follow_action is not None and hasattr(
                    oracle_follow_action, "compute_teacher_base_vel"
                ):
                    raw_action = [
                        float(x)
                        for x in np.asarray(
                            oracle_follow_action.compute_teacher_base_vel(),
                            dtype=np.float64,
                        ).reshape(-1)
                    ]
                else:
                    raw_action = [0.0, 0.0, 0.0]

                track_indices = [1] + _humanoid_agent_indices(sim)
                xz_now = _collect_agent_xz_by_index(sim, track_indices)
                human_prev_xz = (
                    prev_xz_by_agent.get(0)
                    if prev_xz_by_agent is not None
                    else None
                )
                human_velocity_for_guard = _estimate_xz_velocity(
                    xz_now[0], human_prev_xz, ctrl_dt
                )
                action, dyn_corr = _apply_dynamic_avoidance(
                    raw_action,
                    robot_agent,
                    sim,
                    prev_xz_by_agent,
                    ctrl_dt,
                    base_vel_action,
                    velocity_ema_by_agent,
                )
                action, main_guard_corr, main_guard_debug = (
                    _apply_main_human_proximity_guard(
                        action,
                        robot_agent,
                        humanoid_agent_main,
                        base_vel_action,
                        human_velocity_for_guard,
                        ctrl_dt,
                    )
                )
                prev_xz_by_agent = {k: v.copy() for k, v in xz_now.items()}
                robot_config.debug["base_velocity_raw"] = raw_action
                robot_config.debug["dyn_obs_correction"] = dyn_corr
                robot_config.debug["main_human_guard_correction"] = main_guard_corr
                for guard_key, guard_value in main_guard_debug.items():
                    robot_config.debug[f"main_human_guard_{guard_key}"] = guard_value

                action_dict = {
                    "action": _TRACK_COMPOSITE_ACTION_NAMES,
                    "action_args": {"agent_1_base_vel": action},
                }

                iter_step += 1
                env.step(action_dict)

                base_velocity_physical = _physical_base_velocity(action, base_vel_action)
                target_pose = _target_pose_from_base_velocity(action, ctrl_dt)
                info = env.get_metrics()

                robot_pos_after = _get_base_pos(robot_agent)
                human_pos_after = _get_base_pos(humanoid_agent_main)
                robot_yaw_after = _get_base_yaw(robot_agent)
                human_yaw_after = _get_base_yaw(humanoid_agent_main)

                if _measure_scalar(info.get("human_following", 0.0)) >= 1.0 - 1e-6:
                    followed_step += 1
                    too_far_count = 0
                else:
                    if np.linalg.norm(robot_pos_after - human_pos_after) > 4.0:
                        too_far_count += 1
                    if too_far_count > 20:
                        print("Too far from human!")
                        status = "Lost"
                        finished = False
                        break

                frame_index = len(record_infos)
                record_info = _build_step_info_record(
                    frame_index=frame_index,
                    ctrl_dt=ctrl_dt,
                    iter_step=iter_step,
                    robot_pos_before=robot_pos_before,
                    robot_yaw_before=robot_yaw_before,
                    robot_pos_after=robot_pos_after,
                    robot_yaw_after=robot_yaw_after,
                    human_pos_before=human_pos_before,
                    human_yaw_before=human_yaw_before,
                    human_pos_after=human_pos_after,
                    human_yaw_after=human_yaw_after,
                    action=action,
                    base_velocity_physical=base_velocity_physical,
                    target_pose=target_pose,
                    info=info,
                )
                record_info["base_velocity_raw"] = [float(x) for x in raw_action]
                record_info["dyn_obs_correction"] = float(dyn_corr)
                record_info["main_human_guard_correction"] = float(main_guard_corr)
                record_info["main_human_guard_speed_limit"] = float(
                    main_guard_debug["toward_speed_limit"]
                )
                for guard_key, guard_value in main_guard_debug.items():
                    record_info[f"main_human_guard_{guard_key}"] = (
                        bool(guard_value)
                        if isinstance(guard_value, (bool, np.bool_))
                        else float(guard_value)
                    )
                record_infos.append(record_info)
                debug = dict(robot_config.debug)

                if human_stuck_filter:
                    if (
                        _human_xz_step_displacement(human_pos_before, human_pos_after)
                        < human_stuck_eps
                    ):
                        human_stuck_count += 1
                    else:
                        human_stuck_count = 0
                    if human_stuck_count >= human_stuck_min_steps:
                        print(
                            f"Main human stuck for {human_stuck_count} steps "
                            f"(episode {episode_id})!"
                        )
                        status = "HumanStuck"
                        finished = False
                        break

                if info.get("human_collision", 0.0) == 1.0:
                    print("Collision detected!")
                    status = "Collision"
                    finished = False
                    break

                _dbg_mode = debug.get("mode", "-")
                _vis_extra = (
                    f" vis_det:{debug.get('visible_detector')} "
                    f"vis_geom:{debug.get('visible_geom_gt')}"
                )
                if os.environ.get("DATA_QUIET") != "1":
                    print(
                        f"========== ID: {env.current_episode.episode_id} "
                        f"Step: {iter_step} mode:{_dbg_mode} action: {action} "
                        f"dis_to_main_human: {np.linalg.norm(robot_pos_after - human_pos_after):.3f} "
                        f"visible: {debug.get('visible')}{_vis_extra} ============"
                    )

            # print("finished episode id: ", env.current_episode.episode_id)
            info = env.get_metrics()

            if env.episode_over:
                finished = True

            result["finish"] = finished
            result["status"] = status
            _hf_succ = _measure_scalar(info.get("human_following_success", 0.0))

            result["following_rate"] = float(followed_step / max(iter_step, 1))
            result["following_step"] = int(followed_step)
            result["total_step"] = int(iter_step)
            result["collision"] = info.get("human_collision", None)
            result["oracle_robot_teacher"] = True
            result["dyn_obstacle_avoid"] = os.environ.get("DYN_OBSTACLE_AVOID", "1") != "0"
            result["main_human_proximity_guard"] = bool(
                guard_runtime_config["enabled"]
            )
            result["main_human_guard_config"] = dict(guard_runtime_config)
            result["avoid_other_pedestrians"] = _avoid_other_pedestrians_enabled()
            result["avoid_backend"] = "rvo2"
            result["ctrl_freq"] = int(sim.ctrl_freq)
            result["fps"] = int(sim.ctrl_freq)
            result["scene_id"] = scene_key
            result["episode_id"] = episode_id
            result["robot_type"] = "habitat_stretch"
            result["task_type"] = "tracking"
            result["task_subtype"] = "human_following"
            if instruction is not None:
                result["instruction"] = instruction
            result["data_debug"] = debug_save_all
            last_human_following = (
                len(record_infos) > 0
                and _measure_scalar(record_infos[-1].get("human_following", 0.0))
                >= 1.0 - 1e-6
            )
            result["last_human_following"] = bool(last_human_following)
            result["human_following_success"] = bool(_hf_succ >= 1.0 - 1e-6)
            final_visible_detector = bool(
                robot_config.debug.get("visible_detector", False)
            )
            result["success_visible_detector"] = final_visible_detector
            result["success"] = bool(
                (result["human_following_success"] or last_human_following)
                and final_visible_detector
            )

            if debug_save_all:
                save_episode = len(record_infos) > 0
                if save_episode:
                    n_snap = len(record_infos)
                    cached_record_infos = copy.deepcopy(record_infos)
                    cached_video_frames = robot_config.video_frames_snapshot()
                    front_frames = cached_video_frames.get("front", [])
                    if len(front_frames) != len(cached_record_infos):
                        print(
                            f"[WARN] episode {env.current_episode.episode_id} 缓存对齐: "
                            f"front视频帧数={len(front_frames)} record行数={len(cached_record_infos)}"
                        )
            else:
                save_episode = (
                    len(record_infos) > 0
                    and result["success"]
                    and status == "Normal"
                    and result["following_rate"] >= 0.5
                )

                if save_episode:
                    n_snap = len(record_infos)
                    cached_record_infos = copy.deepcopy(record_infos)
                    if human_stuck_filter:
                        stuck_run = _max_consecutive_human_stuck_steps(
                            cached_record_infos, human_stuck_eps
                        )
                        if stuck_run >= human_stuck_min_steps:
                            print(
                                f"[discard] {scene_key}/{episode_id}: episode has "
                                f"{stuck_run} consecutive main-human stuck steps "
                                f"(threshold={human_stuck_min_steps})"
                            )
                            save_episode = False
                            status = "HumanStuck"
                            result["status"] = status
                            result["success"] = False
                    cached_video_frames = robot_config.video_frames_snapshot()
                    front_frames = cached_video_frames.get("front", [])
                    if save_episode and len(front_frames) != len(cached_record_infos):
                        print(
                            f"[WARN] episode {env.current_episode.episode_id} 缓存对齐: "
                            f"front视频帧数={len(front_frames)} record行数={len(cached_record_infos)}"
                        )
            if save_episode:
                result_save = dict(result)
                result_save["total_step"] = n_snap
                result_save["following_step"] = sum(
                    1
                    for r in cached_record_infos
                    if _measure_scalar(r.get("facing", 0.0)) >= 1.0 - 1e-6
                )
                result_save["following_rate"] = float(
                    result_save["following_step"] / max(n_snap, 1)
                )
                result_save["used_success_snapshot"] = False
                result_save["saved_full_episode"] = True
                if base_vel_meta:
                    result_save["agent_1_base_velocity"] = base_vel_meta
                # 保存完整 episode 视频帧。
                robot_config.set_video_frames(cached_video_frames)

                save_dir = os.path.join(save_path, scene_key)
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, f"{env.current_episode.episode_id}_info.json"), "w") as f:
                    json.dump(cached_record_infos, f, indent=2)
                with open(os.path.join(save_dir, f"{env.current_episode.episode_id}.json"), "w") as f:
                    json.dump(result_save, f, indent=2)
            elif status == "HumanStuck":
                discarded_stuck_episodes += 1

            rollout_succeeded = (
                len(record_infos) > 0
                and bool(result.get("success", False))
                and status == "Normal"
                and float(result.get("following_rate", 0.0)) >= 0.5
            )
            if not rollout_succeeded:
                reasons = _rollout_failure_reasons(result, record_infos, status)
                failure_log_path = _append_rollout_failure_log(
                    save_path=save_path,
                    split_id=split_id,
                    scene_key=scene_key,
                    episode_id=episode_id,
                    result=result,
                    record_infos=record_infos,
                    reasons=reasons,
                )
                failed_episodes += 1
                for reason in reasons:
                    failure_reason_counts[reason] = (
                        failure_reason_counts.get(reason, 0) + 1
                    )
                print(
                    f"[rollout-failed] split={split_id} "
                    f"episode={scene_key}/{episode_id} "
                    f"reasons={','.join(reasons)}"
                )

            # save_episode 为 True 时写视频并清空缓存；为 False 时仅清空本局累积的视频帧。
            robot_config.reset(env.current_episode, success=save_episode)

        if skipped_episodes > 0:
            print(
                f"[skip] split skipped {skipped_episodes}/{num_episodes} "
                f"already saved episode(s) under {save_path}"
            )
        if discarded_stuck_episodes > 0:
            print(
                f"[discard] split dropped {discarded_stuck_episodes}/{num_episodes} "
                f"episode(s) due to main-human stuck "
                f"(min_steps={human_stuck_min_steps}, xz_eps={human_stuck_eps})"
            )
        if failed_episodes > 0:
            print(
                f"[rollout-failures] split={split_id} "
                f"failed={failed_episodes}/{num_episodes} "
                f"reason_counts={json.dumps(failure_reason_counts, sort_keys=True)} "
                f"log={failure_log_path}"
            )
