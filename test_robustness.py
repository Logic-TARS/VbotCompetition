#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鲁棒性测试脚本：评估模型在多次测试中的成功率和稳定性
Robustness Testing Script: Evaluate model success rate and stability across multiple trials
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'motrix_envs/src'))

def compute_projected_gravity(quat):
    """计算机器人坐标系中的重力向量"""
    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    if quat.ndim == 1:
        quat = quat[np.newaxis, :]
        gravity_vec = gravity_vec[np.newaxis, :]
    elif quat.ndim == 2:
        gravity_vec = np.tile(gravity_vec, (quat.shape[0], 1))
    
    # 四元数逆旋转
    # q_conj = [qx, qy, qz, -qw] (simplified for unit quaternions)
    qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # 旋转向量 v' = q^-1 * v * q
    # 简化为直接矩阵乘法
    x, y, z = gravity_vec[..., 0], gravity_vec[..., 1], gravity_vec[..., 2]
    
    # 应用四元数旋转（逆旋转）
    rotated_x = (1 - 2*(qy**2 + qz**2)) * x + 2*(qx*qy + qw*qz) * y + 2*(qx*qz - qw*qy) * z
    rotated_y = 2*(qx*qy - qw*qz) * x + (1 - 2*(qx**2 + qz**2)) * y + 2*(qy*qz + qw*qx) * z
    rotated_z = 2*(qx*qz + qw*qy) * x + 2*(qy*qz - qw*qx) * y + (1 - 2*(qx**2 + qy**2)) * z
    
    if quat.ndim == 1:
        return np.array([rotated_x.item(), rotated_y.item(), rotated_z.item()])
    else:
        return np.stack([rotated_x, rotated_y, rotated_z], axis=-1)

def test_robustness(num_trials=10, steps_per_trial=1000):
    """测试机器狗导航任务的鲁棒性
    
    Args:
        num_trials: 测试次数
        steps_per_trial: 每次测试的步数
    
    Returns:
        bool: 是否达到成功率和稳定性目标
    """
    
    print("=" * 80)
    print("VBot 鲁棒性测试")
    print("=" * 80)
    print(f"测试配置：")
    print(f"  - 测试次数: {num_trials}")
    print(f"  - 每次步数: {steps_per_trial}")
    print(f"  - 成功率目标: 70-80%")
    print(f"  - 稳定性目标: 平均倾斜 < 32°")
    print("=" * 80)
    
    try:
        from motrix_envs import registry
        
        # 创建环境
        print("\n[1] 创建环境...")
        env = registry.make("vbot_navigation_section001", "np", num_envs=10)
        print(f"    ✓ 环境创建成功 (10只机器狗)")
        
        success_count = 0
        stability_metrics = []
        fall_count = 0
        out_of_bounds_count = 0
        
        print("\n[2] 开始测试...")
        
        for trial in range(num_trials):
            print(f"\n  试验 {trial + 1}/{num_trials}:")
            
            # 初始化环境
            state = env.init_state()
            trial_max_tilt = 0.0
            trial_fell = False
            trial_succeeded = False
            
            for step in range(steps_per_trial):
                # 使用随机动作（模拟训练早期的探索）
                actions = np.random.uniform(-1.0, 1.0, size=(10, 12)).astype(np.float32)
                
                # 执行步骤
                state = env.step(actions)
                
                # 收集稳定性指标
                if 'root_quat' in state.info:
                    quat = state.info['root_quat']
                    
                    # 计算投影重力
                    gravity = compute_projected_gravity(quat)
                    
                    # 计算倾斜角（基于重力向量的XY分量）
                    # 如果重力完全向下，XY分量应该接近0
                    # 倾斜角 = arcsin(sqrt(gx^2 + gy^2))
                    tilt_magnitude = np.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)
                    tilt_angles = np.arcsin(np.clip(tilt_magnitude, -1.0, 1.0))
                    max_tilt = np.max(tilt_angles)
                    
                    stability_metrics.append(np.rad2deg(max_tilt))
                    
                    if max_tilt > trial_max_tilt:
                        trial_max_tilt = max_tilt
                
                # 检查是否有机器狗摔倒或越界
                if state.terminated is not None and np.any(state.terminated):
                    trial_fell = True
                    fall_count += 1
                    break
                
                # 检查是否完成任务（到达目标）
                if 'total_score' in state.info and state.info['total_score'] >= 2:
                    trial_succeeded = True
                    success_count += 1
                    break
            
            # 输出试验结果
            status = "✓ 成功" if trial_succeeded else ("✗ 摔倒" if trial_fell else "- 超时")
            print(f"    {status} | 最大倾斜: {np.rad2deg(trial_max_tilt):.1f}°")
        
        # 统计结果
        print("\n" + "=" * 80)
        print("测试结果汇总")
        print("=" * 80)
        
        success_rate = success_count / num_trials
        avg_stability = np.mean(stability_metrics) if stability_metrics else 0.0
        max_stability = np.max(stability_metrics) if stability_metrics else 0.0
        
        print(f"✓ 成功次数: {success_count}/{num_trials}")
        print(f"✓ 成功率: {success_rate * 100:.1f}% (目标: 70-80%)")
        print(f"✓ 摔倒次数: {fall_count}")
        print(f"✓ 平均倾斜: {avg_stability:.1f}° (目标: <32°)")
        print(f"✓ 最大倾斜: {max_stability:.1f}°")
        print("=" * 80)
        
        # 判断是否达标
        target_achieved = success_rate >= 0.7 and avg_stability < 32
        
        if target_achieved:
            print("\n🎉 测试通过！达到鲁棒性目标。")
        else:
            print("\n⚠️  测试未通过。需要进一步优化：")
            if success_rate < 0.7:
                print(f"   - 成功率 {success_rate * 100:.1f}% < 70%")
            if avg_stability >= 32:
                print(f"   - 平均倾斜 {avg_stability:.1f}° >= 32°")
        
        return target_achieved
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_domain_randomization():
    """测试域随机化是否正常工作"""
    
    print("\n" + "=" * 80)
    print("域随机化功能测试")
    print("=" * 80)
    
    try:
        from motrix_envs import registry
        from motrix_envs.navigation.vbot.cfg import VBotSection001EnvCfg
        
        # 创建环境
        print("\n[1] 检查配置...")
        env = registry.make("vbot_navigation_section001", "np", num_envs=5)
        cfg = env._cfg
        
        # 检查域随机化配置
        if hasattr(cfg, 'domain_randomization'):
            dr = cfg.domain_randomization
            print("    ✓ 域随机化配置存在")
            print(f"      - 质量范围: {dr.mass_scale_range}")
            print(f"      - 摩擦范围: {dr.friction_scale_range}")
            print(f"      - 关节位置噪声: ±{dr.init_qpos_noise_scale}")
            print(f"      - 关节速度噪声: ±{dr.init_qvel_noise_scale}")
            print(f"      - 随机推力概率: {dr.random_push_prob * 100}%")
            print(f"      - 推力大小: ±{dr.random_push_scale} m/s")
        else:
            print("    ✗ 域随机化配置不存在")
            return False
        
        print("\n[2] 测试初始化多样性...")
        initial_positions = []
        initial_velocities = []
        
        for i in range(10):
            state = env.init_state()
            if 'root_pos' in state.info:
                initial_positions.append(state.info['root_pos'][:, :2])  # XY位置
            if 'root_vel' in state.info:
                initial_velocities.append(np.linalg.norm(state.info['root_vel'][:, :2], axis=1))
        
        if initial_positions:
            all_positions = np.concatenate(initial_positions, axis=0)
            pos_std = np.std(all_positions, axis=0)
            print(f"    ✓ 初始位置标准差: X={pos_std[0]:.3f}m, Y={pos_std[1]:.3f}m")
        
        if initial_velocities:
            all_velocities = np.concatenate(initial_velocities, axis=0)
            vel_mean = np.mean(all_velocities)
            vel_std = np.std(all_velocities)
            print(f"    ✓ 初始速度统计: 均值={vel_mean:.3f}m/s, 标准差={vel_std:.3f}m/s")
        
        print("\n[3] 检查奖励配置...")
        reward_scales = cfg.reward_config.scales
        
        key_rewards = {
            "orientation": -0.20,
            "lin_vel_z": -0.30,
            "ang_vel_xy": -0.15,
            "contact_stability": 0.1,
            "action_smoothness": -0.01,
        }
        
        all_correct = True
        for key, expected_value in key_rewards.items():
            actual_value = reward_scales.get(key, None)
            if actual_value is not None:
                if abs(actual_value - expected_value) < 1e-6:
                    print(f"    ✓ {key}: {actual_value} (正确)")
                else:
                    print(f"    ✗ {key}: {actual_value} (期望: {expected_value})")
                    all_correct = False
            else:
                print(f"    ✗ {key}: 未配置")
                all_correct = False
        
        if all_correct:
            print("\n✓ 域随机化功能测试通过")
        else:
            print("\n✗ 部分配置不正确")
        
        return all_correct
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 首先测试域随机化配置
    print("开始测试...")
    
    dr_passed = test_domain_randomization()
    
    if dr_passed:
        # 然后测试鲁棒性
        # 注意：完整的鲁棒性测试需要训练好的模型
        # 这里使用随机策略进行基本功能验证
        print("\n注意：使用随机策略进行基本功能验证（非性能测试）")
        robustness_passed = test_robustness(num_trials=3, steps_per_trial=100)
    else:
        print("\n⚠️  跳过鲁棒性测试，因为配置测试未通过")
        robustness_passed = False
    
    # 最终结果
    print("\n" + "=" * 80)
    if dr_passed:
        print("✓ 域随机化和奖励函数配置正确")
        print("✓ 环境初始化测试通过")
        print("\n注意：完整的鲁棒性测试需要训练模型后进行")
        print("建议：使用 `uv run scripts/train.py --env vbot_navigation_section001` 训练模型")
    else:
        print("✗ 测试未完全通过，请检查配置")
    print("=" * 80)
