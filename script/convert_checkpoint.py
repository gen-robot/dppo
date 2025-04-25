import os
import torch
import argparse
from pathlib import Path

def print_checkpoint_structure(ckpt, prefix=""):
    """打印检查点结构
    
    Args:
        ckpt: 检查点字典
        prefix: 前缀字符串，用于缩进显示
    """
    if isinstance(ckpt, dict):
        print(f"{prefix}字典包含以下键:")
        for key, value in ckpt.items():
            print(f"{prefix}  - {key}: ", end="")
            if isinstance(value, (dict, torch.Tensor)):
                print()
                print_checkpoint_structure(value, prefix + "    ")
            else:
                print(f"类型 = {type(value)}")
    elif isinstance(ckpt, torch.Tensor):
        print(f"{prefix}张量: shape={ckpt.shape}, dtype={ckpt.dtype}")
    else:
        print(f"{prefix}其他类型: {type(ckpt)}")

def compare_checkpoints(ckpt1_path: str, ckpt2_path: str):
    """比较两个checkpoint的结构
    
    Args:
        ckpt1_path: 第一个checkpoint文件路径
        ckpt2_path: 第二个checkpoint文件路径
    """
    # 加载checkpoint文件
    print(f"正在加载第一个检查点: {ckpt1_path}")
    ckpt1 = torch.load(ckpt1_path, map_location="cpu")
    
    print(f"\n正在加载第二个检查点: {ckpt2_path}")
    ckpt2 = torch.load(ckpt2_path, map_location="cpu")
    
    # 打印检查点结构
    print("\n第一个检查点结构:")
    print_checkpoint_structure(ckpt1)
    
    print("\n第二个检查点结构:")
    print_checkpoint_structure(ckpt2)

def main():
    parser = argparse.ArgumentParser(description="比较两个checkpoint的结构")
    parser.add_argument("--ckpt1-path", type=str, required=True, help="第一个checkpoint文件路径")
    parser.add_argument("--ckpt2-path", type=str, required=True, help="第二个checkpoint文件路径")
    
    args = parser.parse_args()
    compare_checkpoints(args.ckpt1_path, args.ckpt2_path)

if __name__ == "__main__":
    main() 