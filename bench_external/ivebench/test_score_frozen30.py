from __future__ import annotations

import sys
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_frozen30 as frozen30


ROOT = Path(__file__).resolve().parent


class VariantConfigurationTests(unittest.TestCase):
    def test_vace_base_baseline_has_no_lora_and_pins_vace_defaults(self) -> None:
        sys.path.insert(0, str(ROOT.parents[1] / "src"))
        import edit
        from edit.video_editor import EDIT_VARIANTS

        variant = EDIT_VARIANTS["wan14b-vace-base"]
        self.assertEqual(variant.backend, "ditto")
        self.assertEqual(variant.lora_paths, ())
        self.assertEqual(variant.num_inference_steps, 50)
        self.assertEqual(variant.cfg_scale, 5.0)


class Frozen30SelectionTests(unittest.TestCase):
    def test_selection_is_balanced_and_score_independent(self) -> None:
        styles = frozen30.read_jsonl(ROOT / "ive_direct.jsonl")
        attrs = frozen30.read_jsonl(ROOT / "ive_prompts_attr.jsonl")
        picked_style = frozen30.stratified_evenly(styles, 15)
        picked_attr = [
            row
            for category, quota in frozen30.ATTR_QUOTAS.items()
            for row in frozen30.evenly_spaced(
                [r for r in attrs if r["subcategory"] == category], quota
            )
        ]

        self.assertEqual(len(picked_style), 15)
        self.assertEqual(len(picked_attr), 15)
        counts = Counter(row["subcategory"] for row in picked_attr)
        self.assertEqual(counts, frozen30.ATTR_QUOTAS)
        style_counts = Counter(row["subcategory"] for row in picked_style)
        self.assertLessEqual(max(style_counts.values()) - min(style_counts.values()), 1)
        self.assertEqual(
            len({frozen30.video_name(row) for row in picked_style + picked_attr}), 30
        )

    def test_exact_sign_test_handles_ties_and_symmetric_samples(self) -> None:
        self.assertEqual(frozen30.sign_test_p([0.0, 0.0]), 1.0)
        self.assertEqual(frozen30.sign_test_p([1.0, -1.0]), 1.0)
        self.assertAlmostEqual(frozen30.sign_test_p([1.0] * 5), 0.0625)

    def test_extra_evaluation_arm_and_remote_csv_fallback(self) -> None:
        arms = frozen30.parse_arm_specs(
            ["Bernini=eval_bernini"], [("direct", "eval_direct"), ("skill", "eval_skill")]
        )
        self.assertEqual(arms["Bernini"], "eval_bernini")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "eval_bernini"
            folder.mkdir()
            (folder / "summary.json").write_text(json.dumps({"csv": "/data/expired.csv"}))
            csv_path = folder / "bernini_eval_results.csv"
            headers = ["video_name", *frozen30.METRICS]
            values = ["short_0001.mp4", "0.1", "0.2", "0.9", "0.97"]
            csv_path.write_text(",".join(headers) + "\n" + ",".join(values) + "\n")

            scores = frozen30.load_scores(root, arms["Bernini"])
            self.assertAlmostEqual(
                scores["short_0001.mp4"]["phrase_semantic_consistency_score"], 0.2
            )

    def test_full_report_pairs_multiple_methods_by_video_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            style_rows = []
            attr_rows = []
            for i in range(48):
                sid = f"short_{i + 1:04d}"
                style_rows.append({
                    "sample_id": sid, "instruction": "restyle", "category": "style_editing",
                    "subcategory": f"style_{i % 8}", "source_prompt": "source",
                    "target_prompt": "target", "target_span": "style",
                    "target_phrase": "style",
                })
            for i in range(40):
                sid = f"short_{i + 101:04d}"
                subcategory = tuple(frozen30.ATTR_QUOTAS)[i % 3]
                attr_rows.append({
                    "sample_id": sid, "src_video_name": f"{sid}.mp4",
                    "instruction": "edit", "category": "attribute_editing",
                    "subcategory": subcategory, "source_prompt": "source",
                    "target_prompt": "target", "target_span": "object",
                    "target_phrase": "object",
                })
            for filename, rows in (
                ("ive_direct.jsonl", style_rows),
                ("ive_prompts_attr.jsonl", attr_rows),
                ("ive_skill.jsonl", [
                    {**row, "instruction": "compiled style"}
                    for row in style_rows
                ]),
                ("ive_attr_skill.jsonl", [
                    {**row, "instruction": "compiled point edit"}
                    for row in attr_rows
                ]),
            ):
                (root / filename).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )

            style_names = [f"{row['sample_id']}.mp4" for row in style_rows]
            attr_names = [row["src_video_name"] for row in attr_rows]
            metric_names = list(frozen30.METRICS)
            for folder_name, names, base in (
                ("eval_direct", style_names, 0.1),
                ("eval_skill", style_names, 0.2),
                ("eval_vace", style_names, 0.3),
                ("eval_attr_direct", attr_names, 0.1),
                ("eval_attr_skill", attr_names, 0.2),
                ("eval_attr_vace", attr_names, 0.3),
            ):
                folder = root / folder_name
                folder.mkdir()
                (folder / "summary.json").write_text(
                    json.dumps({"csv": "/data/not-mounted.csv"})
                )
                with (folder / "scores_eval_results.csv").open("w") as handle:
                    handle.write(",".join(["video_name", *metric_names]) + "\n")
                    for name in names:
                        handle.write(",".join([name, *[str(base)] * 4]) + "\n")

            report_path = root / "scorecard.md"
            argv = [
                "score_frozen30.py", "--root", str(root),
                "--style-arm", "VACE=eval_vace",
                "--point-arm", "VACE=eval_attr_vace",
                "--report-out", str(report_path),
            ]
            with patch.object(sys, "argv", argv):
                frozen30.main()
            report = report_path.read_text()
            self.assertIn("| PSC | VACE | 0.3000 | +0.2000", report)
            self.assertIn("Skill−method (95% CI)", report)
            self.assertIn("style=VACE, point_edit=VACE", report)
            style_skill = [
                json.loads(line)
                for line in (root / "FROZEN30_STYLE_SKILL.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(style_skill), 15)
            self.assertTrue(all(row["instruction"] == "compiled style" for row in style_skill))
            point_skill = [
                json.loads(line)
                for line in (root / "FROZEN30_POINT_EDIT_SKILL.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(point_skill), 15)
            self.assertTrue(all(row["instruction"] == "compiled point edit" for row in point_skill))

            style_rows[0]["instruction"] = "silently changed benchmark prompt"
            (root / "ive_direct.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in style_rows)
            )
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                ValueError, "no longer match frozen manifest"
            ):
                frozen30.main()


if __name__ == "__main__":
    unittest.main()
