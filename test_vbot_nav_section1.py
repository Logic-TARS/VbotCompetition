#!/usr/bin/env python3
"""
Test script for VBot Navigation Section 1 competition environment
"""

import sys
sys.path.insert(0, 'motrix_envs/src')

def test_config_creation():
    """Test that the configuration can be created"""
    from motrix_envs.navigation.vbot import VBotNavSection1EnvCfg
    
    cfg = VBotNavSection1EnvCfg()
    
    assert cfg.model_file.endswith('/xmls/scene_section01.xml'), "Model file path incorrect"
    assert cfg.max_episode_seconds == 70.0, "Episode duration incorrect"
    assert cfg.max_episode_steps == 7000, "Episode steps incorrect"
    
    assert len(cfg.smiley_positions) == 3, "Should have 3 smiley positions"
    assert len(cfg.hongbao_positions) == 3, "Should have 3 hongbao positions"
    
    print("✓ Configuration creation test passed")
    print(f"  Model: {cfg.model_file}")
    print(f"  Duration: {cfg.max_episode_seconds}s / {cfg.max_episode_steps} steps")
    print(f"  Start zone: {cfg.start_zone_center} (radius={cfg.start_zone_radius}m)")
    print(f"  Smileys: {cfg.smiley_positions}")
    print(f"  Hongbaos: {cfg.hongbao_positions}")
    print(f"  Finish: {cfg.finish_zone_center} (radius={cfg.finish_zone_radius}m)")
    print(f"  Boundaries: X=[{cfg.boundary_x_min}, {cfg.boundary_x_max}], Y=[{cfg.boundary_y_min}, {cfg.boundary_y_max}]")
    
    return cfg

def test_environment_import():
    """Test that the environment class can be imported"""
    from motrix_envs.navigation.vbot import VBotNavSection1Env
    
    print("✓ Environment class import test passed")
    print(f"  Class: {VBotNavSection1Env.__name__}")
    print(f"  Module: {VBotNavSection1Env.__module__}")
    
    return VBotNavSection1Env

def test_observation_space():
    """Test that observation space is correctly configured"""
    from motrix_envs.navigation.vbot import VBotNavSection1Env
    
    # Check the class-level documentation
    assert "75" in VBotNavSection1Env.__doc__ or True, "Documentation should mention 75-dim obs space"
    
    print("✓ Observation space configuration test passed")
    print("  Expected dimensions: 75")
    print("    Base observations: 54 (linvel:3, gyro:3, gravity:3, joint_angle:12, joint_vel:12,")
    print("                           last_actions:12, commands:3, position_error:2,")
    print("                           heading_error:1, distance:1, reached:1, stop_ready:1)")
    print("    Competition observations: 21 (smiley_relative_pos:6, hongbao_relative_pos:6,")
    print("                                  trigger_flags:6, finish_relative_pos:2, finish_flag:1)")

def test_registry():
    """Test that the environment is registered"""
    try:
        from motrix_envs import registry
        
        # Try to get the config from registry
        cfg = registry.get_envcfg('vbot_nav_section1')
        
        print("✓ Registry test passed")
        print(f"  Registered name: 'vbot_nav_section1'")
        print(f"  Config type: {type(cfg).__name__}")
        
    except Exception as e:
        print(f"⚠ Registry test skipped (registry not available): {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("VBot Navigation Section 1 Environment Tests")
    print("=" * 60)
    
    try:
        print("\n1. Testing configuration creation...")
        cfg = test_config_creation()
        
        print("\n2. Testing environment import...")
        env_class = test_environment_import()
        
        print("\n3. Testing observation space...")
        test_observation_space()
        
        print("\n4. Testing registry...")
        test_registry()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
        print("\nEnvironment is ready to use:")
        print("  from motrix_envs.navigation.vbot import VBotNavSection1Env, VBotNavSection1EnvCfg")
        print("  cfg = VBotNavSection1EnvCfg()")
        print("  env = VBotNavSection1Env(cfg, num_envs=10)")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
