#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Velocity Trap Fix Configuration Validation

This script validates the configuration changes made to fix the zero-velocity local optimum.
It only checks configuration values without initializing the environment.
"""

import sys
import os

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'motrix_envs/src'))

def test_reward_config():
    """Test that reward configuration has been correctly updated"""
    print("=" * 80)
    print("Testing Reward Configuration")
    print("=" * 80)
    
    from motrix_envs.navigation.vbot.cfg import VBotSection001EnvCfg
    
    cfg = VBotSection001EnvCfg()
    scales = cfg.reward_config.scales
    
    tests_passed = 0
    tests_failed = 0
    
    # Define expected values
    expected = {
        "forward_velocity": 2.0,
        "orientation": -0.05,
        "lin_vel_z": -0.10,
        "ang_vel_xy": -0.05,
        "foot_air_time": 0.3,
        "action_smoothness": 0.1,
        "contact_stability": 0.2,
        "torques": -0.00001,
        "action_rate": -0.001,
    }
    
    for key, expected_val in expected.items():
        if key not in scales:
            print(f"❌ {key}: NOT FOUND")
            tests_failed += 1
        elif scales[key] == expected_val:
            print(f"✓ {key}: {scales[key]}")
            tests_passed += 1
        else:
            print(f"❌ {key}: expected {expected_val}, got {scales[key]}")
            tests_failed += 1
    
    print(f"\nReward Config: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0

def test_config_parameters():
    """Test that new configuration parameters exist"""
    print("\n" + "=" * 80)
    print("Testing Configuration Parameters")
    print("=" * 80)
    
    from motrix_envs.navigation.vbot.cfg import VBotSection001EnvCfg
    
    cfg = VBotSection001EnvCfg()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test force_initial_motion parameter
    if hasattr(cfg, 'force_initial_motion'):
        if cfg.force_initial_motion == True:
            print(f"✓ force_initial_motion: {cfg.force_initial_motion}")
            tests_passed += 1
        else:
            print(f"❌ force_initial_motion: expected True, got {cfg.force_initial_motion}")
            tests_failed += 1
    else:
        print(f"❌ force_initial_motion: NOT FOUND")
        tests_failed += 1
    
    # Test recovery_tilt_threshold parameter
    if hasattr(cfg, 'recovery_tilt_threshold'):
        if cfg.recovery_tilt_threshold == 80.0:
            print(f"✓ recovery_tilt_threshold: {cfg.recovery_tilt_threshold}°")
            tests_passed += 1
        else:
            print(f"❌ recovery_tilt_threshold: expected 80.0, got {cfg.recovery_tilt_threshold}")
            tests_failed += 1
    else:
        print(f"❌ recovery_tilt_threshold: NOT FOUND")
        tests_failed += 1
    
    print(f"\nConfig Parameters: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0

def main():
    """Run all validation tests"""
    print("\n" + "=" * 80)
    print("ZERO-VELOCITY TRAP FIX - CONFIGURATION VALIDATION")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: Reward configuration
    try:
        passed = test_reward_config()
        all_passed = all_passed and passed
    except Exception as e:
        print(f"\n❌ Reward configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test 2: Configuration parameters
    try:
        passed = test_config_parameters()
        all_passed = all_passed and passed
    except Exception as e:
        print(f"\n❌ Configuration parameters test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if all_passed:
        print("✅ ALL CONFIGURATION TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
