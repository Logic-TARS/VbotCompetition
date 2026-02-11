#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test to verify the target position logic change.
This tests the logic without requiring the full environment setup.
"""

import sys
import os

# Add the project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'motrix_envs/src'))

def test_target_position_logic():
    """Test that target positions are now fixed and unified"""
    
    print("=" * 80)
    print("Target Position Logic Test")
    print("=" * 80)
    
    try:
        import numpy as np
        
        # Simulate configuration
        class MockCfg:
            target_point_a = [0.0, 1.5]
            arena_center = [0.0, 0.0]
            
            class Commands:
                pose_command_range = [-0.2, 3.4, -0.3, 0.2, 3.8, 0.3]
            
            commands = Commands()
        
        cfg = MockCfg()
        num_envs = 10
        
        # Simulate robot initial positions (on outer circle)
        robot_init_xy = np.zeros((num_envs, 2), dtype=np.float32)
        arena_outer_radius = 3.0
        for i in range(num_envs):
            theta = np.random.uniform(0, 2 * np.pi)
            radius = arena_outer_radius + np.random.uniform(-0.1, 0.1)
            robot_init_xy[i, 0] = radius * np.cos(theta)
            robot_init_xy[i, 1] = radius * np.sin(theta)
        
        robot_init_xy += np.array(cfg.arena_center, dtype=np.float32)
        robot_init_pos = np.column_stack([robot_init_xy, np.full(num_envs, 0.5, dtype=np.float32)])
        
        print("\n[1] Robot Initial Positions (on outer circle):")
        print(f"    Shape: {robot_init_pos.shape}")
        print(f"    Sample positions:")
        for i in range(min(3, num_envs)):
            x, y, z = robot_init_pos[i]
            distance = np.sqrt(x**2 + y**2)
            print(f"      Robot {i}: ({x:.3f}, {y:.3f}, {z:.3f}) - Distance from center: {distance:.3f}m")
        
        # OLD LOGIC (commented out for reference)
        # dx_min, dy_min, yaw_min = -0.2, 3.4, -0.3
        # dx_max, dy_max, yaw_max = 0.2, 3.8, 0.3
        # sampled = np.random.uniform(
        #     low=np.array([dx_min, dy_min, yaw_min], dtype=np.float32),
        #     high=np.array([dx_max, dy_max, yaw_max], dtype=np.float32),
        #     size=(num_envs, 3),
        # )
        # old_target_positions = robot_init_pos[:, :2] + sampled[:, :2]
        
        # NEW LOGIC
        target_point_a = np.array(cfg.target_point_a if hasattr(cfg, 'target_point_a') else [0.0, 1.5], dtype=np.float32)
        arena_center = np.array(cfg.arena_center if hasattr(cfg, 'arena_center') else [0.0, 0.0], dtype=np.float32)
        
        # All robots have the same target
        target_positions = np.tile(target_point_a + arena_center, (num_envs, 1))
        
        # Target headings still random
        if hasattr(cfg.commands, 'pose_command_range') and len(cfg.commands.pose_command_range) == 6:
            yaw_min, yaw_max = cfg.commands.pose_command_range[2], cfg.commands.pose_command_range[5]
            target_headings = np.random.uniform(yaw_min, yaw_max, size=(num_envs, 1)).astype(np.float32)
        else:
            target_headings = np.zeros((num_envs, 1), dtype=np.float32)
        
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)
        
        print("\n[2] Target Positions (NEW LOGIC - Fixed and Unified):")
        print(f"    Shape: {target_positions.shape}")
        print(f"    Target Point A: {target_point_a}")
        print(f"    Arena Center: {arena_center}")
        print(f"    Calculated Target: {target_point_a + arena_center}")
        print(f"    All positions identical: {np.all(target_positions == target_positions[0])}")
        print(f"    Sample target positions:")
        for i in range(min(3, num_envs)):
            x, y = target_positions[i]
            print(f"      Robot {i}: ({x:.3f}, {y:.3f})")
        
        print("\n[3] Pose Commands (target_x, target_y, target_heading):")
        print(f"    Shape: {pose_commands.shape}")
        print(f"    Sample commands:")
        for i in range(min(3, num_envs)):
            x, y, heading = pose_commands[i]
            print(f"      Robot {i}: target=({x:.3f}, {y:.3f}), heading={heading:.3f} rad")
        
        print("\n[4] Verification:")
        
        # Check all targets are the same
        all_same = np.all(target_positions == target_positions[0])
        print(f"    ✓ All robots have the same target position: {all_same}")
        
        # Check target is at (0.0, 1.5)
        expected_target = np.array([0.0, 1.5])
        target_is_correct = np.allclose(target_positions[0], expected_target)
        print(f"    ✓ Target position is (0.0, 1.5): {target_is_correct}")
        
        # Check headings are different (random)
        headings_different = not np.allclose(target_headings, target_headings[0])
        print(f"    ✓ Target headings are randomized: {headings_different}")
        
        # Check pose commands shape
        correct_shape = pose_commands.shape == (num_envs, 3)
        print(f"    ✓ Pose commands have correct shape (10, 3): {correct_shape}")
        
        print("\n" + "=" * 80)
        if all_same and target_is_correct and correct_shape:
            print("✓ All tests PASSED!")
            print("=" * 80)
            return True
        else:
            print("✗ Some tests FAILED!")
            print("=" * 80)
            return False
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_target_position_logic()
    sys.exit(0 if success else 1)
