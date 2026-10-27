"""Verify fixed-image COCO scoring and truthful compilation metadata."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_coco import score_coco, validate_detections
from inspect_compilation import layer_group_info


class CocoTests(unittest.TestCase):

    def test_category_ids_are_dataset_ids_and_nonfinite_results_fail(self):
        valid = {"image_id": 1, "category_id": 3, "bbox": [0, 0, 4, 5], "score": 0.9}
        validate_detections([valid], {1}, {1, 3})
        with self.assertRaises(ValueError):
            validate_detections([{**valid, "category_id": 2}], {1}, {1, 3})
        with self.assertRaises(ValueError):
            validate_detections([{**valid, "score": float("nan")}], {1}, {1, 3})

    def test_images_without_predictions_remain_in_evaluation(self):
        try:
            from pycocotools.coco import COCO
        except ImportError:
            self.skipTest("pycocotools is optional")
        ground_truth = COCO()
        ground_truth.dataset = {
            "info": {},
            "images": [{"id": 1, "width": 20, "height": 20},
                       {"id": 2, "width": 20, "height": 20}],
            "categories": [{"id": 3, "name": "fixture"}],
            "annotations": [{"id": i, "image_id": i, "category_id": 3,
                             "bbox": [1, 2, 5, 6], "area": 30, "iscrowd": 0} for i in (1, 2)]}
        ground_truth.createIndex()
        predictions = [{"image_id": i, "category_id": 3, "bbox": [1, 2, 5, 6],
                        "score": 0.9} for i in (1, 2)]
        full = score_coco(ground_truth, predictions, [1, 2])["AP_50_95"]
        partial = score_coco(ground_truth, predictions[:1], [1, 2])["AP_50_95"]
        empty = score_coco(ground_truth, [], [1, 2])["AP_50_95"]
        self.assertAlmostEqual(full, 1.0)
        self.assertLess(partial, 0.75)
        self.assertEqual(empty, 0.0)


class GroupMetadataTests(unittest.TestCase):

    def test_real_ginfo_scalar_and_slice_fields(self):
        info = layer_group_info("#tpu.lg<out_addr = 8192, out_size = 512, buffer_addr = 16384, "
                                "buffer_size = 256, id = 3, stage = 1, "
                                "h_idx = [0, 15], h_slice = [17, 16], w_idx = [], w_slice = []>")
        self.assertEqual(info["h_idx"], [0, 15])
        self.assertEqual(info["h_slice"], [17, 16])
        self.assertEqual(info["w_slice"], [])
        self.assertEqual(info["out_addr"], 8192)
        self.assertNotIn("lmem_live_peak_bytes", info)


if __name__ == "__main__":
    unittest.main()
