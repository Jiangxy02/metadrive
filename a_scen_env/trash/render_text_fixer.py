"""
渲染文本修复工具 - 解决MetaDrive渲染中的dtype错误
"""

import numpy as np
from typing import Dict, Any, Union


def safe_str_convert(value: Any) -> str:
    """安全地将任意类型转换为字符串"""
    if value is None:
        return "N/A"
    
    # 处理NumPy数组
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return str(float(value))
        else:
            return str(value.tolist())
    
    # 处理NumPy标量
    if isinstance(value, (np.integer, np.floating)):
        return str(float(value))
    
    # 处理列表/元组
    if isinstance(value, (list, tuple)):
        try:
            if len(value) == 2 and all(isinstance(x, (int, float, np.number)) for x in value):
                return f"({float(value[0]):.1f}, {float(value[1]):.1f})"
            else:
                return str(value)
        except:
            return str(value)
    
    # 处理布尔值
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    
    # 处理数值类型
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.3f}"
        else:
            return str(value)
    
    return str(value)


def fix_render_text(text_dict: Dict[str, Any]) -> Dict[str, str]:
    """修复渲染文本字典，确保所有值都是字符串"""
    if not isinstance(text_dict, dict):
        return {}
    
    fixed_dict = {}
    
    for key, value in text_dict.items():
        safe_key = safe_str_convert(key)
        safe_value = safe_str_convert(value)
        fixed_dict[safe_key] = safe_value
    
    return fixed_dict 