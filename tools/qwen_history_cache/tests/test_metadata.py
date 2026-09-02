"""Tests for deterministic EVT metadata joins."""

import pyarrow as pa
import pytest
from qwen_history_cache.metadata import assemble_worklist


def _tables() -> tuple[pa.Table, pa.Table, pa.Table]:
    refs = pa.table(
        {
            "ref_id": [0, 1],
            "ref": ["episode/000000/front", "episode/000001/front"],
            "episode_id": ["episode", "episode"],
            "frame_index": [0, 1],
            "camera_name": ["front", "front"],
        }
    )
    records = pa.table(
        {
            "episode_id": ["episode", "episode"],
            "frame_index": [0, 1],
            "data_index": [10, 11],
        }
    )
    video = pa.table(
        {
            "data_index": [10, 11],
            "video_key": ["front_image", "front_image"],
            "video_frame_index": [4, 5],
            "chunk_index": [2, 2],
            "file_index": [7, 7],
        }
    )
    return refs, records, video


def test_assemble_worklist_preserves_ref_order_and_paths() -> None:
    refs, records, video = _tables()
    result = assemble_worklist(refs, records, video).to_pylist()
    assert [row["ref_id"] for row in result] == [0, 1]
    assert result[0]["data_index"] == 10
    assert result[0]["video_path"] == "videos/front_image/chunk-002/part-007.mp4"


def test_assemble_worklist_rejects_missing_join() -> None:
    refs, records, video = _tables()
    with pytest.raises(ValueError, match="front-video join"):
        assemble_worklist(refs, records, video.slice(0, 1))


def test_assemble_worklist_rejects_duplicate_source_key() -> None:
    refs, records, video = _tables()
    duplicate = pa.concat_tables([records, records.slice(0, 1)])
    with pytest.raises(ValueError, match="unique"):
        assemble_worklist(refs, duplicate, video)


def test_assemble_worklist_rejects_nonfront_ref() -> None:
    refs, records, video = _tables()
    refs = refs.set_column(
        refs.schema.get_field_index("camera_name"),
        "camera_name",
        pa.array(["front", "left"]),
    )
    with pytest.raises(ValueError, match="front-camera"):
        assemble_worklist(refs, records, video)
