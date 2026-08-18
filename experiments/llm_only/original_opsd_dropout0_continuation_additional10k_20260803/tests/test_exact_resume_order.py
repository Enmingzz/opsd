import json
import random

from opsd.visionzip_aokvqa.aokvqa import load_aokvqa_dataset


def _write_rows(path, count=12):
    rows = []
    for index in range(count):
        row = {
            "sample_id": f"sample-{index:02d}",
            "image": f"image-{index:02d}.jpg",
            "question": f"Question {index}?",
            "answer": "<think>Reasoning.</think>\n<answer>yes</answer>",
        }
        rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def test_shuffle_false_preserves_preordered_jsonl(tmp_path):
    path = tmp_path / "ordered.jsonl"
    rows = _write_rows(path)

    loaded = load_aokvqa_dataset(str(path), seed=42, shuffle=False)

    assert [sample.sample_id for sample in loaded] == [row["sample_id"] for row in rows]


def test_default_shuffle_behavior_is_unchanged(tmp_path):
    path = tmp_path / "ordered.jsonl"
    rows = _write_rows(path)
    expected = [row["sample_id"] for row in rows]
    random.Random(42).shuffle(expected)

    loaded_default = load_aokvqa_dataset(str(path), seed=42)
    loaded_explicit = load_aokvqa_dataset(str(path), seed=42, shuffle=True)

    assert [sample.sample_id for sample in loaded_default] == expected
    assert [sample.sample_id for sample in loaded_explicit] == expected
