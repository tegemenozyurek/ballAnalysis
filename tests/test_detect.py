from __future__ import annotations

import unittest

import numpy as np

from live_football_vision.detect import Detection, count_labels, parse_detections
from live_football_vision.model import BALL_CLASS_ID, PLAYER_CLASS_ID


class ParseDetectionsTests(unittest.TestCase):
    def test_maps_coco_classes_and_clips_boxes(self) -> None:
        xyxy = np.array(
            [
                [-10, -4, 40, 80],
                [10, 10, 30, 30],
                [5, 5, 50, 50],
            ],
            dtype=float,
        )
        class_ids = np.array([PLAYER_CLASS_ID, BALL_CLASS_ID, 17])
        confidences = np.array([0.91, 0.44, 0.99])

        detections = parse_detections(xyxy, class_ids, confidences, 64, 48)

        self.assertEqual(
            detections,
            (
                Detection(label="player", confidence=0.91, x1=0, y1=0, x2=40, y2=47),
                Detection(label="ball", confidence=0.44, x1=10, y1=10, x2=30, y2=30),
            ),
        )

    def test_counts_players_and_balls(self) -> None:
        detections = (
            Detection("player", 0.9, 0, 0, 10, 10),
            Detection("player", 0.8, 12, 0, 20, 10),
            Detection("ball", 0.5, 30, 30, 36, 36),
        )
        self.assertEqual(count_labels(detections), (2, 1))


if __name__ == "__main__":
    unittest.main()
