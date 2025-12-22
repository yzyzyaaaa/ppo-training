#!/usr/bin/env python3
"""
使用训练好的模型进行推理测试并生成效果视频
"""
import os
import sys
import torch
import numpy as np
import gymnasium as gym
import imageio
import cv2

sys.path.insert(0, '/mnt/RL/yzy/RLinf')
from rlinf.models.embodiment.mlp_policy import MLPPolicy
from rlinf.hybrid_engines.fsdp.strategy.checkpoint import load_checkpoint

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def add_text_to_frame(frame, text, position=(20, 40), color=(0, 255, 0), font_scale=1.0):
    """在帧上添加文字"""
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    cv2.rectangle(frame, 
                  (position[0]-5, position[1]-text_size[1]-5),
                  (position[0]+text_size[0]+5, position[1]+5),
                  (0, 0, 0), -1)
    
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def load_trained_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"加载检查点: {checkpoint_path}")
    
    # 模型配置（与训练配置一致）
    obs_dim = 45  # PickSingleYCB环境的观测维度
    action_dim = 8
    hidden_dim = 256
    num_action_chunks = 1
    add_value_head = True
    
    # 创建模型
    model = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_action_chunks=num_action_chunks,
        add_value_head=add_value_head,
    )
    
    # 加载检查点
    if os.path.isdir(checkpoint_path):
        # FSDP检查点目录
        actor_dir = os.path.join(checkpoint_path, "actor")
        if os.path.exists(actor_dir):
            # 使用distcp格式加载
            checkpoint_files = [f for f in os.listdir(actor_dir) if f.endswith('.distcp')]
            if checkpoint_files:
                # 尝试加载distcp格式的检查点
                # 注意：这里需要根据实际的FSDP检查点格式来加载
                print(f"找到检查点文件: {checkpoint_files[0]}")
                # 由于FSDP检查点的复杂格式，我们需要使用RLinf的加载函数
                try:
                    # 尝试直接使用torch加载（如果格式兼容）
                    checkpoint = torch.load(os.path.join(actor_dir, checkpoint_files[0]), map_location='cpu')
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # 移除FSDP包装前缀
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_key = k.replace('_fsdp_wrapped_module.', '').replace('module.', '')
                        new_state_dict[new_key] = v
                    
                    model.load_state_dict(new_state_dict, strict=False)
                    print("模型加载成功（使用distcp格式）")
                except Exception as e:
                    print(f"加载distcp格式失败: {e}")
                    print("尝试使用FSDP加载方式...")
                    # 如果直接加载失败，使用评估框架来加载
                    return None
    else:
        # 直接加载pth文件
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("模型加载成功（使用pth格式）")
    
    model.cuda()
    model.eval()
    return model

def main():
    print("="*70)
    print("使用训练好的模型进行推理测试")
    print("="*70)
    
    # 检查点路径
    checkpoint_path = "/mnt/RL/yzy/recordings/rlinf_logs/pickplace_multishape_mlp/checkpoints/global_step_1200"
    
    # 由于FSDP检查点格式复杂，我们使用评估框架的方式
    # 这里创建一个简化的推理脚本，使用评估配置
    print("\n使用RLinf评估框架运行推理...")
    print("这将加载模型并生成10个episodes的效果视频")
    
    # 直接运行评估脚本来生成视频
    import subprocess
    
    eval_script = "/mnt/RL/yzy/run_eval_pickplace.sh"
    if os.path.exists(eval_script):
        print(f"运行评估脚本: {eval_script}")
        result = subprocess.run(["bash", eval_script], capture_output=False)
        if result.returncode == 0:
            print("\n评估完成！")
            # 复制最新的视频到目标位置
            eval_video_dir = "/mnt/RL/yzy/recordings/eval_videos/seed_42"
            if os.path.exists(eval_video_dir):
                # 找到最新的视频
                video_files = [f for f in os.listdir(eval_video_dir) if f.endswith('.mp4')]
                if video_files:
                    # 按修改时间排序
                    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(eval_video_dir, x)), reverse=True)
                    latest_video = os.path.join(eval_video_dir, video_files[0])
                    target_video = "/mnt/RL/yzy/recordings/2.mp4"
                    import shutil
                    shutil.copy2(latest_video, target_video)
                    print(f"视频已保存到: {target_video}")
                    print(f"视频大小: {os.path.getsize(target_video) / 1024 / 1024:.2f} MB")
        else:
            print("评估脚本执行失败")
            return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


