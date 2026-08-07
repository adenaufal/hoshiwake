from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorter import (  # noqa: E402
    _resolve_collision,
    determine_category,
    discover_images,
    load_image,
    score_groups,
    sort_file,
)


class ScoreGroupsTests(unittest.TestCase):
    def test_binary_sfw_nsfw_labels_are_not_double_counted(self):
        sfw, nsfw = score_groups({"sfw": 0.11, "nsfw": 0.89})
        self.assertAlmostEqual(sfw, 0.11)
        self.assertAlmostEqual(nsfw, 0.89)

    def test_safe_unsafe_labels_are_not_double_counted(self):
        sfw, nsfw = score_groups({"safe": 0.10, "unsafe": 0.90})
        self.assertAlmostEqual(sfw, 0.10)
        self.assertAlmostEqual(nsfw, 0.90)

    def test_siglip2_exact_labels_aggregate(self):
        sfw, nsfw = score_groups(
            {
                "Anime Picture": 0.4,
                "Normal": 0.3,
                "Hentai": 0.2,
                "Pornography": 0.05,
                "Enticing or Sensual": 0.05,
            }
        )
        self.assertAlmostEqual(sfw, 0.7)
        self.assertAlmostEqual(nsfw, 0.25)

    def test_mixed_exact_and_keyword_labels(self):
        # "Normal" matches exactly, "Explicit" only via keyword; both must count.
        sfw, nsfw = score_groups({"Normal": 0.01, "Explicit": 0.99})
        self.assertAlmostEqual(sfw, 0.01)
        self.assertAlmostEqual(nsfw, 0.99)

    def test_caveduck_allow_prohibit_labels(self):
        sfw, nsfw = score_groups({"allow": 0.2, "prohibit": 0.8})
        self.assertAlmostEqual(sfw, 0.2)
        self.assertAlmostEqual(nsfw, 0.8)

    def test_fallback_index_labels(self):
        sfw, nsfw = score_groups({"label_0": 0.3, "label_1": 0.7})
        self.assertAlmostEqual(sfw, 0.3)
        self.assertAlmostEqual(nsfw, 0.7)

    def test_unrelated_labels_count_for_neither_group(self):
        sfw, nsfw = score_groups({"cat": 0.5, "dog": 0.5})
        self.assertEqual((sfw, nsfw), (0.0, 0.0))

    def test_spelled_out_negated_labels_are_nsfw(self):
        for nsfw_label in ("not_safe_for_work", "not-safe-for-work", "not safe for work", "notsafeforwork"):
            sfw, nsfw = score_groups({"safe_for_work": 0.05, nsfw_label: 0.95})
            self.assertAlmostEqual(sfw, 0.05, msg=nsfw_label)
            self.assertAlmostEqual(nsfw, 0.95, msg=nsfw_label)

    def test_open_nsfw_style_five_class_labels(self):
        sfw, nsfw = score_groups(
            {"Neutral": 0.5, "Drawings": 0.3, "Sexy": 0.05, "Hentai": 0.1, "Porn": 0.05}
        )
        self.assertAlmostEqual(sfw, 0.8)
        self.assertAlmostEqual(nsfw, 0.15)

    def test_index_fallback_labels_match_exactly_not_by_substring(self):
        # "label_1" must not match inside "label_12".
        sfw, nsfw = score_groups({"label_12": 1.0})
        self.assertEqual((sfw, nsfw), (0.0, 0.0))

    def test_non_numeric_scores_are_skipped(self):
        sfw, nsfw = score_groups({"sfw": None, "nsfw": 0.9, "other": "bad"})
        self.assertEqual(sfw, 0.0)
        self.assertAlmostEqual(nsfw, 0.9)


class DetermineCategoryTests(unittest.TestCase):
    def test_binary_nsfw_model_can_return_nsfw(self):
        result = {"all_scores": {"sfw": 0.11, "nsfw": 0.89}}
        self.assertEqual(determine_category(result, 0.65, 0.10), "NSFW")

    def test_binary_sfw_model_returns_sfw(self):
        result = {"all_scores": {"sfw": 0.92, "nsfw": 0.08}}
        self.assertEqual(determine_category(result, 0.65, 0.10), "SFW")

    def test_below_threshold_is_uncertain(self):
        result = {"all_scores": {"sfw": 0.55, "nsfw": 0.45}}
        self.assertEqual(determine_category(result, 0.65, 0.10), "UNCERTAIN")

    def test_within_margin_is_uncertain(self):
        result = {"all_scores": {"sfw": 0.52, "nsfw": 0.48}}
        self.assertEqual(determine_category(result, 0.5, 0.10), "UNCERTAIN")

    def test_normal_explicit_pair_reaches_nsfw(self):
        result = {"all_scores": {"Normal": 0.01, "Explicit": 0.99}}
        self.assertEqual(determine_category(result, 0.65, 0.10), "NSFW")

    def test_negated_safe_label_reaches_nsfw(self):
        result = {"all_scores": {"safe_for_work": 0.05, "not_safe_for_work": 0.95}}
        self.assertEqual(determine_category(result, 0.65, 0.10), "NSFW")

    def test_neutral_dominant_five_class_reaches_sfw(self):
        result = {"all_scores": {"neutral": 0.7, "drawings": 0.2, "sexy": 0.02, "hentai": 0.05, "porn": 0.03}}
        self.assertEqual(determine_category(result, 0.65, 0.10), "SFW")

    def test_single_label_path_nsfw(self):
        result = {"label": "nsfw", "score": 0.85}
        self.assertEqual(determine_category(result, 0.65, 0.10), "NSFW")

    def test_single_label_path_below_threshold(self):
        result = {"label": "nsfw", "score": 0.5}
        self.assertEqual(determine_category(result, 0.65, 0.10), "UNCERTAIN")

    def test_empty_result_is_uncertain(self):
        self.assertEqual(determine_category({}, 0.65, 0.10), "UNCERTAIN")


class SortFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_copy_creates_destination(self):
        src = self.tmp / "img.png"
        src.write_bytes(b"data")
        out = self.tmp / "out"
        destination = sort_file(src, out, "SFW", "copy")
        self.assertEqual(destination, out / "SFW" / "img.png")
        self.assertTrue(destination.exists())
        self.assertTrue(src.exists())

    def test_move_removes_source(self):
        src = self.tmp / "img.png"
        src.write_bytes(b"data")
        out = self.tmp / "out"
        destination = sort_file(src, out, "NSFW", "move")
        self.assertTrue(destination.exists())
        self.assertFalse(src.exists())

    def test_collision_appends_counter(self):
        src = self.tmp / "img.png"
        src.write_bytes(b"new")
        out = self.tmp / "out"
        existing = out / "SFW" / "img.png"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"old")
        destination = sort_file(src, out, "SFW", "copy")
        self.assertEqual(destination.name, "img_1.png")
        self.assertEqual(existing.read_bytes(), b"old")

    def test_self_copy_is_a_noop(self):
        # File already sitting in the destination category folder.
        out = self.tmp / "out"
        src = out / "SFW" / "img.png"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"data")
        destination = sort_file(src, out, "SFW", "copy")
        self.assertEqual(destination, src)
        self.assertEqual(len(list(src.parent.iterdir())), 1)

    def test_unsupported_mode_raises(self):
        src = self.tmp / "img.png"
        src.write_bytes(b"data")
        with self.assertRaises(ValueError):
            sort_file(src, self.tmp / "out", "SFW", "delete")

    def test_resolve_collision_returns_free_name(self):
        target = self.tmp / "img.png"
        self.assertEqual(_resolve_collision(target), target)
        target.write_bytes(b"x")
        self.assertEqual(_resolve_collision(target), self.tmp / "img_1.png")


class DiscoverAndLoadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_discover_filters_extensions_and_sorts(self):
        (self.tmp / "b.png").write_bytes(b"")
        (self.tmp / "a.jpg").write_bytes(b"")
        (self.tmp / "notes.txt").write_bytes(b"")
        (self.tmp / "sub").mkdir()
        names = [p.name for p in discover_images(self.tmp)]
        self.assertEqual(names, ["a.jpg", "b.png"])

    def test_load_image_returns_none_for_corrupt_file(self):
        corrupt = self.tmp / "bad.png"
        corrupt.write_bytes(b"not an image at all")
        self.assertIsNone(load_image(corrupt))

    def test_load_image_returns_rgb(self):
        from PIL import Image

        path = self.tmp / "ok.png"
        Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(path)
        image = load_image(path)
        self.assertIsNotNone(image)
        self.assertEqual(image.mode, "RGB")
        image.close()


if __name__ == "__main__":
    unittest.main()
