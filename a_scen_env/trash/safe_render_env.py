"""
安全渲染环境基类 - 自动修复所有渲染文本错误

用法：
    class YourEnv(SafeRenderEnv, SomeOtherEnv):
        # 你的环境实现
        pass
"""

from metadrive.a_scen_env.trash.render_text_fixer import fix_render_text


class SafeRenderMixin:
    """
    安全渲染混入类
    
    自动修复MetaDrive环境中的渲染文本错误
    "the resolved dtypes are not compatible with add.reduce"
    """
    
    def render(self, *args, **kwargs):
        """
        重写render方法，自动修复文本参数
        """
        try:
            # 修复text参数
            if 'text' in kwargs and kwargs['text'] is not None:
                kwargs['text'] = fix_render_text(kwargs['text'])
            
            # 调用父类的render方法
            return super().render(*args, **kwargs)
            
        except Exception as e:
            error_msg = str(e)
            
            # 检查是否是dtype错误
            if "dtype" in error_msg.lower() or "add.reduce" in error_msg:
                print(f"渲染错误，尝试修复: {error_msg}")
                
                # 尝试1: 清空文本再渲染
                if 'text' in kwargs:
                    kwargs['text'] = {}
                    try:
                        return super().render(*args, **kwargs)
                    except:
                        pass
                
                # 尝试2: 完全移除文本参数
                kwargs.pop('text', None)
                try:
                    return super().render(*args, **kwargs)
                except Exception as e2:
                    print(f"渲染完全失败: {e2}")
                    # 返回None或默认值
                    return None
            else:
                # 不是dtype错误，重新抛出
                raise e


def make_env_safe_render(env_class):
    """
    装饰器：使任何环境类具有安全渲染能力
    
    用法:
        @make_env_safe_render
        class MyEnv(MetaDriveEnv):
            pass
    """
    
    class SafeRenderEnv(SafeRenderMixin, env_class):
        """动态创建的安全渲染环境类"""
        pass
    
    # 保持原有类名
    SafeRenderEnv.__name__ = f"SafeRender{env_class.__name__}"
    SafeRenderEnv.__qualname__ = f"SafeRender{env_class.__qualname__}"
    
    return SafeRenderEnv


# === 使用示例 ===
if __name__ == "__main__":
    print("安全渲染环境基类示例:")
    print("1. 混入类用法:")
    print("   class YourEnv(SafeRenderMixin, MetaDriveEnv):")
    print("       pass")
    print()
    print("2. 装饰器用法:")
    print("   @make_env_safe_render")
    print("   class YourEnv(MetaDriveEnv):")
    print("       pass") 