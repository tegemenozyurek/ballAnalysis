from __future__ import annotations

import unittest

from live_football_vision.region import Region
from live_football_vision.region_select import _map_display_box_to_region


class RegionTests(unittest.TestCase):
    def test_rejects_non_positive_size(self) -> None:
        with self.assertRaises(ValueError):
            Region(0, 0, 0, 100)
        with self.assertRaises(ValueError):
            Region(0, 0, 100, -1)

    def test_mss_dict(self) -> None:
        region = Region(left=10, top=20, width=1280, height=720)
        self.assertEqual(
            region.to_mss(),
            {"left": 10, "top": 20, "width": 1280, "height": 720},
        )


class RegionMappingTests(unittest.TestCase):
    def test_retina_full_scale_display(self) -> None:
        screen = Region(left=0, top=0, width=1512, height=982)
        mapped = _map_display_box_to_region(
            200, 100, 400, 300, screen, display_scale=1.0, pixel_scale=2.0
        )
        self.assertEqual(mapped, Region(left=100, top=50, width=100, height=100))

    def test_negative_monitor_origin(self) -> None:
        screen = Region(left=-1920, top=0, width=1920, height=1080)
        mapped = _map_display_box_to_region(
            0, 0, 1920, 1080, screen, display_scale=1.0, pixel_scale=1.0
        )
        self.assertEqual(mapped, Region(left=-1920, top=0, width=1920, height=1080))


if __name__ == "__main__":
    unittest.main()
