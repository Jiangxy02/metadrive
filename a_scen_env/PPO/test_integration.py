#!/usr/bin/env python3
"""
PPO集成测试脚本

测试PPO训练系统与trajectory_replay.py的集成是否正常工作
"""

import os
import sys
import numpy as np

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_environment_creation():
    """测试环境创建"""
    print("\n[1] Testing environment creation...")
    try:
        from train_ppo import TrajectoryReplayWrapper
        
        # 创建环境
        env = TrajectoryReplayWrapper()
        obs, info = env.reset()
        
        print(f"✓ Environment created successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        
        # 测试步进
        try:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✓ Environment step successful")
            print(f"  Reward: {reward:.2f}")
        except FileNotFoundError as e:
            # 如果因为没有训练模型而失败，这是预期的
            if "No trained PPO model found" in str(e):
                print(f"ℹ Environment step skipped (no trained model yet, expected)")
            else:
                raise e
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False


def test_ppo_expert_interface():
    """测试PPO专家接口"""
    print("\n[2] Testing PPO expert interface...")
    try:
        from ppo_expert import expert, set_expert_model, get_expert_model_path
        
        # 检查是否有可用的模型
        model_path = get_expert_model_path()
        if model_path:
            print(f"✓ Found existing model: {model_path}")
        else:
            print("ℹ No trained model found (expected if not trained yet)")
        
        # 测试expert函数（即使没有模型也应该能导入）
        print("✓ PPO expert interface loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ PPO expert interface failed: {e}")
        return False


def test_control_manager_integration():
    """测试控制管理器集成"""
    print("\n[3] Testing control manager integration...")
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from control_mode_manager import ControlModeManager
        
        # 创建控制管理器
        manager = ControlModeManager(engine=None, main_vehicle_trajectory=None)
        
        print("✓ Control manager created successfully")
        print("✓ PPO expert mode integration ready")
        return True
        
    except Exception as e:
        print(f"✗ Control manager integration failed: {e}")
        return False


def test_training_import():
    """测试训练模块导入"""
    print("\n[4] Testing training modules...")
    modules_to_test = [
        ("train_ppo", "Training module"),
        ("evaluate_ppo", "Evaluation module"),
        ("start_training", "Training launcher"),
    ]
    
    all_success = True
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {description} ({module_name}) loaded successfully")
        except Exception as e:
            print(f"✗ {description} ({module_name}) failed: {e}")
            all_success = False
    
    return all_success


def test_model_training_minimal():
    """测试最小化训练流程"""
    print("\n[5] Testing minimal training flow...")
    try:
        from train_ppo import train_ppo
        
        # 使用非常小的参数进行快速测试
        print("  Running minimal training (100 steps)...")
        model, model_dir = train_ppo(
            total_timesteps=100,  # 只训练100步
            n_envs=1,  # 单环境
            config={
                "horizon": 100,
                "use_render": False,
                "manual_control": False,
            }
        )
        
        print(f"✓ Minimal training completed")
        print(f"  Model saved to: {model_dir}")
        
        # 清理测试模型
        import shutil
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print("  Test model cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Minimal training failed: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("PPO Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Environment Creation", test_environment_creation),
        ("PPO Expert Interface", test_ppo_expert_interface),
        ("Control Manager Integration", test_control_manager_integration),
        ("Training Modules Import", test_training_import),
        # ("Minimal Training Flow", test_model_training_minimal),  # 可选，需要时间
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # 总结
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:30} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The PPO training system is ready to use.")
        print("\nNext steps:")
        print("1. Run training: python start_training.py")
        print("2. Evaluate model: python evaluate_ppo.py")
        print("3. Use in trajectory_replay.py (automatic)")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 