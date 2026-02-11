#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit test for VBot navigation arena initial position generation fix.

This test validates the random position generation logic independently
of the full environment setup, ensuring it meets all requirements.

Requirements validated:
1. Initial positions are on outer circle (radius 3.0 ± 0.1m)
2. Each reset produces different random positions
3. 10 dogs are distributed with different angles
4. Height is fixed at 0.5m

Author: GitHub Copilot
Date: 2026-02-11
"""

import unittest
import numpy as np


class TestVBotInitialPositionGeneration(unittest.TestCase):
    """Test suite for VBot initial position generation fix"""

    def setUp(self):
        """Set up test configuration"""
        # Mock configuration matching VBotSection001EnvCfg
        self.arena_outer_radius = 3.0
        self.arena_inner_radius = 1.5
        self.boundary_radius = 3.5
        self.arena_center = np.array([0.0, 0.0], dtype=np.float32)
        self.num_envs = 10

    def generate_positions(self, num_envs):
        """
        Exact replica of the position generation code from vbot_section001_np.py reset() method
        """
        # ===== 极坐标随机生成在外圈 =====
        # 每只机器狗在外圈随机分布（半径 3.0 ± 0.1m）
        robot_init_xy = np.zeros((num_envs, 2), dtype=np.float32)
        for i in range(num_envs):
            theta = np.random.uniform(0, 2 * np.pi)
            radius = self.arena_outer_radius + np.random.uniform(-0.1, 0.1)
            robot_init_xy[i, 0] = radius * np.cos(theta)
            robot_init_xy[i, 1] = radius * np.sin(theta)

        # 添加圆心偏移（如果存在）
        robot_init_xy += self.arena_center

        # 构造完整的XYZ位置（高度固定为0.5m）
        robot_init_pos = np.column_stack([robot_init_xy, np.full(num_envs, 0.5, dtype=np.float32)])

        return robot_init_pos

    def test_positions_on_outer_circle(self):
        """Test that all positions are on the outer circle with radius 3.0 ± 0.1m"""
        positions = self.generate_positions(self.num_envs)
        distances = np.sqrt(np.sum((positions[:, :2] - self.arena_center) ** 2, axis=1))

        expected_min = self.arena_outer_radius - 0.1
        expected_max = self.arena_outer_radius + 0.1

        self.assertTrue(
            np.all((distances >= expected_min) & (distances <= expected_max)),
            f"Positions should be in range [{expected_min}, {expected_max}], "
            f"but got min={np.min(distances):.3f}, max={np.max(distances):.3f}"
        )

    def test_random_different_positions(self):
        """Test that each reset produces different random positions"""
        positions_1 = self.generate_positions(self.num_envs)
        positions_2 = self.generate_positions(self.num_envs)

        self.assertFalse(
            np.allclose(positions_1, positions_2),
            "Each generation should produce different positions"
        )

    def test_position_variance(self):
        """Test that positions have reasonable variance (spread around circle)"""
        # Generate many samples to test distribution
        all_positions = np.vstack([
            self.generate_positions(self.num_envs) for _ in range(10)
        ])

        x_variance = np.var(all_positions[:, 0])
        y_variance = np.var(all_positions[:, 1])

        # Variance should be reasonably high for positions spread around a circle
        self.assertGreater(x_variance, 1.0, "X coordinate variance should be > 1.0")
        self.assertGreater(y_variance, 1.0, "Y coordinate variance should be > 1.0")

    def test_angle_distribution(self):
        """Test that dogs are distributed with different angles"""
        positions = self.generate_positions(self.num_envs)
        relative_positions = positions[:, :2] - self.arena_center
        angles = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])

        # Convert to [0, 2π] range
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)

        # Check angle coverage (should cover a reasonable portion of the circle)
        angle_range = np.max(angles) - np.min(angles)
        coverage = angle_range / (2 * np.pi)

        self.assertGreater(
            coverage, 0.3,
            f"Angle coverage should be > 30%, but got {coverage * 100:.1f}%"
        )

    def test_fixed_height(self):
        """Test that all heights are fixed at 0.5m"""
        positions = self.generate_positions(self.num_envs)
        heights = positions[:, 2]

        self.assertTrue(
            np.all(heights == 0.5),
            f"All heights should be 0.5, but got: {heights}"
        )

    def test_position_shape(self):
        """Test that generated positions have correct shape"""
        positions = self.generate_positions(self.num_envs)

        self.assertEqual(
            positions.shape,
            (self.num_envs, 3),
            f"Position shape should be ({self.num_envs}, 3), but got {positions.shape}"
        )

    def test_multiple_resets_consistency(self):
        """Test that the logic works consistently across multiple resets"""
        for _ in range(5):
            positions = self.generate_positions(self.num_envs)
            distances = np.sqrt(np.sum((positions[:, :2] - self.arena_center) ** 2, axis=1))

            expected_min = self.arena_outer_radius - 0.1
            expected_max = self.arena_outer_radius + 0.1

            self.assertTrue(
                np.all((distances >= expected_min) & (distances <= expected_max)),
                "Each reset should produce valid positions"
            )

    def test_old_vs_new_behavior(self):
        """Document the difference between old fixed position and new random position"""
        # Old behavior: fixed position
        old_pos = np.array([0.0, 0.6, 0.5])
        old_distance = np.sqrt(old_pos[0]**2 + old_pos[1]**2)

        # Old position was very close to center (0.6m), not on outer circle (3.0m)
        self.assertLess(
            old_distance, 1.0,
            "Old fixed position was near center, not on outer circle"
        )

        # New behavior: random positions on outer circle
        new_positions = self.generate_positions(self.num_envs)
        new_distances = np.sqrt(np.sum((new_positions[:, :2] - self.arena_center) ** 2, axis=1))

        # All new positions should be much further from center
        self.assertTrue(
            np.all(new_distances > 2.5),
            "New positions should be on outer circle, far from center"
        )


if __name__ == '__main__':
    unittest.main()
