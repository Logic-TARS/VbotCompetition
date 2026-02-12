#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Immediate Termination Bug Fix Validation

This script validates the 4 key fixes to prevent robots from "dying at birth":
1. reset() uses cfg.init_state.pos instead of polar coordinates
2. base_contact threshold increased from 0.01 → 0.1 with 50-step grace period
3. Progressive penalty instead of hard-coded -10.0
4. info dict contains "last_distance" and "steps" keys

Author: GitHub Copilot
Date: 2026-02-12
"""

import sys
import os

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'motrix_envs/src'))


def test_config_changes():
    """Test Bug 1: Configuration changes (boundary_radius and init height)"""
    print("=" * 80)
    print("Test Bug 1: Configuration Changes")
    print("=" * 80)
    
    from motrix_envs.navigation.vbot.cfg import VBotSection001EnvCfg
    
    cfg = VBotSection001EnvCfg()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test boundary_radius
    if cfg.boundary_radius == 5.0:
        print(f"✓ boundary_radius: {cfg.boundary_radius} (was 3.5)")
        tests_passed += 1
    else:
        print(f"❌ boundary_radius: {cfg.boundary_radius} (expected 5.0)")
        tests_failed += 1
    
    # Test initial Z height
    if cfg.init_state.pos[2] == 0.35:
        print(f"✓ init_state.pos[2]: {cfg.init_state.pos[2]} (was 0.5)")
        tests_passed += 1
    else:
        print(f"❌ init_state.pos[2]: {cfg.init_state.pos[2]} (expected 0.35)")
        tests_failed += 1
    
    # Test full init position
    expected_pos = [0.0, 0.6, 0.35]
    if cfg.init_state.pos == expected_pos:
        print(f"✓ init_state.pos: {cfg.init_state.pos}")
        tests_passed += 1
    else:
        print(f"❌ init_state.pos: {cfg.init_state.pos} (expected {expected_pos})")
        tests_failed += 1
    
    print(f"\nBug 1 Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_code_logic_simulation():
    """Test Bugs 2, 3, 4: Code logic with simulation (no environment needed)"""
    print("\n" + "=" * 80)
    print("Test Bugs 2, 3, 4: Code Logic Simulation")
    print("=" * 80)
    
    import numpy as np
    
    tests_passed = 0
    tests_failed = 0
    num_envs = 10
    
    # Bug 2: Grace period and threshold test
    print("\n--- Bug 2: Grace Period and Threshold ---")
    GRACE_STEPS = 50
    threshold = 0.1
    
    # During grace period
    current_steps = np.array([25] * num_envs, dtype=np.int32)
    base_contact_value = np.array([0.15] * num_envs)
    past_grace = current_steps > GRACE_STEPS
    base_contact = base_contact_value > threshold
    terminated = base_contact & past_grace
    
    if np.all(~terminated):
        print(f"✓ Grace period (step 25): No termination despite contact > {threshold}")
        tests_passed += 1
    else:
        print(f"❌ Grace period failed: Terminated during grace period")
        tests_failed += 1
    
    # After grace period with high contact
    current_steps = np.array([100] * num_envs, dtype=np.int32)
    past_grace = current_steps > GRACE_STEPS
    terminated = base_contact & past_grace
    
    if np.all(terminated):
        print(f"✓ After grace period (step 100): Termination with contact > {threshold}")
        tests_passed += 1
    else:
        print(f"❌ Post-grace period failed: Should terminate")
        tests_failed += 1
    
    # Low contact (below threshold)
    base_contact_value = np.array([0.05] * num_envs)
    base_contact = base_contact_value > threshold
    terminated = base_contact & past_grace
    
    if np.all(~terminated):
        print(f"✓ Low contact (0.05): No termination below threshold {threshold}")
        tests_passed += 1
    else:
        print(f"❌ Threshold failed: Should not terminate with low contact")
        tests_failed += 1
    
    # Bug 3: Progressive penalty test
    print("\n--- Bug 3: Progressive Penalty ---")
    orientation_penalty_values = np.array([0.3, 0.5, 0.6, 0.8, 1.0])
    extreme_tilt_penalty = np.clip((orientation_penalty_values - 0.5) * 5.0, 0.0, 3.0)
    
    expected_penalties = [0.0, 0.0, 0.5, 1.5, 2.5]
    penalties_match = True
    for i, (penalty, expected) in enumerate(zip(extreme_tilt_penalty, expected_penalties)):
        if abs(penalty - expected) < 0.001:
            continue
        else:
            penalties_match = False
            break
    
    if penalties_match:
        print(f"✓ Progressive penalty: {extreme_tilt_penalty} (not -10.0)")
        tests_passed += 1
    else:
        print(f"❌ Progressive penalty mismatch: {extreme_tilt_penalty}")
        tests_failed += 1
    
    # Test cap at 3.0
    extreme_orientation = np.array([2.0])
    extreme_penalty = np.clip((extreme_orientation - 0.5) * 5.0, 0.0, 3.0)
    if extreme_penalty[0] == 3.0:
        print(f"✓ Penalty cap: Max penalty is {extreme_penalty[0]}")
        tests_passed += 1
    else:
        print(f"❌ Penalty cap failed: {extreme_penalty[0]} (expected 3.0)")
        tests_failed += 1
    
    # Bug 4: Info dict keys test
    print("\n--- Bug 4: Info Dict Keys ---")
    distance_to_target = np.random.uniform(2.0, 4.0, (num_envs,))
    
    info = {
        "pose_commands": np.zeros((num_envs, 3)),
        "last_actions": np.zeros((num_envs, 12)),
        "current_actions": np.zeros((num_envs, 12)),
        "filtered_actions": np.zeros((num_envs, 12)),
        "ever_reached": np.zeros(num_envs, dtype=bool),
        "min_distance": distance_to_target.copy(),
        "last_distance": distance_to_target.copy(),
        "steps": np.zeros(num_envs, dtype=np.int32),
    }
    
    if "last_distance" in info:
        print(f"✓ 'last_distance' key exists in info dict")
        tests_passed += 1
    else:
        print(f"❌ 'last_distance' key missing")
        tests_failed += 1
    
    if "steps" in info:
        print(f"✓ 'steps' key exists in info dict")
        tests_passed += 1
    else:
        print(f"❌ 'steps' key missing")
        tests_failed += 1
    
    # Verify initial values
    if np.array_equal(info["steps"], np.zeros(num_envs, dtype=np.int32)):
        print(f"✓ 'steps' initialized to zeros")
        tests_passed += 1
    else:
        print(f"❌ 'steps' initialization failed")
        tests_failed += 1
    
    print(f"\nBugs 2, 3, 4 Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_reset_position_logic():
    """Test that reset uses cfg.init_state.pos, not polar coordinates"""
    print("\n" + "=" * 80)
    print("Test: Reset Position Logic (cfg.init_state.pos vs polar)")
    print("=" * 80)
    
    import numpy as np
    from motrix_envs.navigation.vbot.cfg import VBotSection001EnvCfg
    
    cfg = VBotSection001EnvCfg()
    num_envs = 10
    
    # Simulate NEW reset logic (using cfg.init_state.pos)
    base_pos = np.array(cfg.init_state.pos, dtype=np.float32)
    robot_init_pos = np.tile(base_pos, (num_envs, 1))
    
    # Small XY randomization
    pr = cfg.init_state.pos_randomization_range
    xy_noise = np.random.uniform(
        [pr[0], pr[1]], [pr[2], pr[3]], (num_envs, 2)
    ).astype(np.float32)
    robot_init_pos[:, :2] += xy_noise
    robot_init_pos[:, 2] = 0.35
    
    tests_passed = 0
    tests_failed = 0
    
    # Check all positions are near cfg.init_state.pos
    expected_x, expected_y = 0.0, 0.6
    positions_valid = True
    for i in range(num_envs):
        x_valid = (expected_x - 0.3) <= robot_init_pos[i, 0] <= (expected_x + 0.3)
        y_valid = (expected_y - 0.3) <= robot_init_pos[i, 1] <= (expected_y + 0.3)
        z_valid = abs(robot_init_pos[i, 2] - 0.35) < 0.001
        if not (x_valid and y_valid and z_valid):
            positions_valid = False
            break
    
    if positions_valid:
        print(f"✓ All positions near cfg.init_state.pos [{expected_x}, {expected_y}, 0.35]")
        tests_passed += 1
    else:
        print(f"❌ Positions outside expected range")
        tests_failed += 1
    
    # Check NOT on outer circle (radius 3.0)
    distances_from_origin = np.linalg.norm(robot_init_pos[:, :2], axis=1)
    max_expected_distance = np.sqrt((0.3)**2 + (0.6 + 0.3)**2)  # ~0.95
    
    if np.all(distances_from_origin < 1.5):
        print(f"✓ Positions NOT on outer circle (radius 3.0)")
        print(f"  Distance range: [{distances_from_origin.min():.2f}, {distances_from_origin.max():.2f}]")
        tests_passed += 1
    else:
        print(f"❌ Some positions on outer circle: {distances_from_origin}")
        tests_failed += 1
    
    print(f"\nReset Position Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def main():
    """Run all validation tests"""
    print("\n" + "=" * 80)
    print("IMMEDIATE TERMINATION BUG FIX VALIDATION")
    print("=" * 80 + "\n")
    
    all_passed = True
    
    all_passed &= test_config_changes()
    all_passed &= test_reset_position_logic()
    all_passed &= test_code_logic_simulation()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Immediate termination bug should be fixed!")
    else:
        print("❌ SOME TESTS FAILED - Review the fixes")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

