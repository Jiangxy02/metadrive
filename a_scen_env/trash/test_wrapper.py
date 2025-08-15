#!/usr/bin/env python3
"""
测试MetaDriveRandomWrapper
"""

def test_wrapper():
    """测试包装器是否工作正常"""
    print("🔧 测试MetaDriveRandomWrapper...")
    
    try:
        from train_ppo_with_random_scenarios import MetaDriveRandomWrapper
        
        # 创建包装器环境
        env = MetaDriveRandomWrapper(
            scenario_type="random",
            num_scenarios=10,
            seed=42,
            use_render=False,
            manual_control=False,
            horizon=100
        )
        
        print("✅ 包装器创建成功")
        print(f"观测空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        
        # 测试reset
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
            
        print("✅ 包装器reset成功")
        print(f"观测形状: {obs.shape}")
        
        # 测试step
        action = [0.0, 0.0]
        obs, reward, terminated, truncated, info = env.step(action)
        print("✅ 包装器step成功")
        print(f"奖励: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 包装器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 包装器测试")
    print("=" * 40)
    
    success = test_wrapper()
    
    print("\n" + "=" * 40)
    print(f"结果: {'✅ 成功' if success else '❌ 失败'}") 