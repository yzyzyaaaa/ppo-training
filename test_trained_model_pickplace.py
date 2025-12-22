#!/usr/bin/env python3
"""
使用训练好的模型进行推理测试并生成效果视频（PickSingleYCB环境）
"""
import os
import sys
import torch
import numpy as np
import gymnasium as gym
import imageio
import cv2
import torch.distributed as dist
from torch.distributed.checkpoint import load

sys.path.insert(0, '/mnt/RL/yzy/RLinf')
from rlinf.models.embodiment.mlp_policy import MLPPolicy
import mani_skill.envs  # 注册ManiSkill环境

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29501'

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

def load_model_from_checkpoint(checkpoint_dir):
    """从FSDP检查点加载模型"""
    print(f"从检查点加载模型: {checkpoint_dir}")
    
    # 初始化分布式环境（单进程）
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
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
    actor_dir = os.path.join(checkpoint_dir, "actor")
    if not os.path.exists(actor_dir):
        print(f"错误: 检查点目录不存在: {actor_dir}")
        return None
    
    try:
        # 准备state_dict结构
        state_dict = {'fsdp_checkpoint': {'model': model.state_dict()}}
        
        # 使用DCP加载
        load(state_dict, checkpoint_id=actor_dir)
        
        # 提取模型权重
        model_state = state_dict['fsdp_checkpoint']['model']
        
        # 加载到模型
        model.load_state_dict(model_state)
        print("✓ 模型加载成功!")
        
        # 移动到GPU
        model.cuda()
        model.eval()
        
        # 清理分布式环境
        if dist.is_initialized():
            dist.destroy_process_group()
        
        return model
        
    except Exception as e:
        print(f"加载失败: {e}")
        import traceback
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        return None

def main():
    print("="*70)
    print("使用训练好的模型进行推理测试 - PickSingleYCB环境")
    print("="*70)
    
    # 检查点路径
    checkpoint_path = "/mnt/RL/yzy/recordings/rlinf_logs/pickplace_multishape_mlp/checkpoints/global_step_1200"
    
    # 加载模型
    print("\n1. 加载训练好的模型...")
    model = load_model_from_checkpoint(checkpoint_path)
    if model is None:
        print("模型加载失败，退出")
        return False
    
    print(f"  ✓ 模型加载成功 ({sum(p.numel() for p in model.parameters()):,} 参数)")
    
    # 创建环境
    print("\n2. 创建ManiSkill环境 (PickSingleYCB-v1)...")
    env = gym.make(
        "PickSingleYCB-v1",
        obs_mode="state",
        control_mode=None,
        render_mode="rgb_array",
        num_envs=1,
        max_episode_steps=100,
        sim_backend="gpu",
        sim_config={
            "sim_freq": 100,
            "control_freq": 20,
        },
        sensor_configs={
            "shader_pack": "default",
            "width": 640,
            "height": 480,
        }
    )
    print("  ✓ 环境创建成功")
    
    # 录制视频
    print("\n3. 开始录制10个episodes...")
    print("-"*70)
    
    all_frames = []
    success_count = 0
    total_steps = []
    total_rewards = []
    
    num_episodes = 10
    
    with torch.no_grad():
        for episode in range(num_episodes):
            obs, info = env.reset(seed=episode+42)
            episode_frames = []
            episode_reward = 0.0
            episode_success = False
            steps = 0
            
            print(f"\nEpisode {episode + 1}/{num_episodes}:", end=" ", flush=True)
            
            for step in range(100):  # max_episode_steps
                # 渲染
                frame = env.render()
                if frame is not None:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    if len(frame.shape) == 4:
                        frame = frame[0]
                    
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
                    episode_frames.append(frame_bgr)
                
                # 提取状态
                if isinstance(obs, dict):
                    if 'states' in obs:
                        state = obs['states']
                    elif 'agent' in obs:
                        if 'qpos' in obs['agent']:
                            state = obs['agent']['qpos']
                        else:
                            state = list(obs['agent'].values())[0]
                    else:
                        # 尝试从obs中提取所有数值
                        state_list = []
                        for key, value in obs.items():
                            if isinstance(value, (torch.Tensor, np.ndarray)):
                                val = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                                state_list.append(val.flatten())
                        state = np.concatenate(state_list) if state_list else None
                else:
                    state = obs
                
                # 转换为numpy
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                if hasattr(state, 'flatten'):
                    state = state.flatten()
                else:
                    state = np.array(state).flatten()
                
                # 确保维度正确 (45维)
                if len(state) < 45:
                    state = np.pad(state, (0, 45 - len(state)))
                elif len(state) > 45:
                    state = state[:45]
                
                # 转换为tensor并使用模型预测
                state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
                
                # 准备输入格式（与训练时一致）
                env_obs = {"states": state_tensor}
                
                # 使用模型预测动作
                action_mean = model.actor_mean(state_tensor)
                action = action_mean.squeeze(0).cpu().numpy()
                
                # 裁剪到合法范围
                action = np.clip(action, -1.0, 1.0)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 提取奖励
                if isinstance(reward, (torch.Tensor, np.ndarray)):
                    reward_val = float(reward.item() if hasattr(reward, 'item') else reward[0] if len(reward) > 0 else reward)
                else:
                    reward_val = float(reward)
                
                episode_reward += reward_val
                steps = step + 1
                
                # 检查是否完成
                done = terminated or truncated
                if isinstance(done, (torch.Tensor, np.ndarray)):
                    done = bool(done.item() if hasattr(done, 'item') else done[0] if len(done) > 0 else done)
                else:
                    done = bool(done)
                
                # 检查成功
                if isinstance(info, dict):
                    success_info = info.get('success', False)
                    if isinstance(success_info, (torch.Tensor, np.ndarray)):
                        episode_success = bool(success_info.item() if hasattr(success_info, 'item') else success_info[0] if len(success_info) > 0 else success_info)
                    else:
                        episode_success = bool(success_info)
                
                if done:
                    break
            
            # 统计
            total_steps.append(steps)
            total_rewards.append(episode_reward)
            if episode_success:
                success_count += 1
            
            status = "✓ 成功!" if episode_success else "✗ 未成功"
            print(f"步数: {steps}, 奖励: {episode_reward:.3f}, {status}")
            
            # 添加文字到帧
            episode_text = f"Episode {episode + 1}/10 - TRAINED MODEL"
            if episode_success:
                episode_text += " - SUCCESS!"
                text_color = (0, 255, 0)  # 绿色
            else:
                episode_text += " - INCOMPLETE"
                text_color = (0, 165, 255)  # 橙色
            
            for idx, frame in enumerate(episode_frames):
                frame_with_text = frame.copy()
                add_text_to_frame(frame_with_text, episode_text, (20, 40), text_color, 0.9)
                add_text_to_frame(frame_with_text, f"Step: {idx+1}/{len(episode_frames)}", 
                                (20, 80), (255, 255, 255), 0.7)
                add_text_to_frame(frame_with_text, f"Reward: {episode_reward:.2f}", 
                                (20, 120), (255, 200, 0), 0.7)
                all_frames.append(frame_with_text)
            
            # 分隔帧
            if episode < num_episodes - 1:
                black_frame = np.zeros_like(episode_frames[0])
                add_text_to_frame(black_frame, "--- Next Episode ---", (180, 240), (255, 255, 255), 1.2)
                all_frames.extend([black_frame] * 20)
    
    env.close()
    
    # 保存视频
    print("\n" + "-"*70)
    output_path = "/mnt/RL/yzy/recordings/2.mp4"
    print(f"4. 保存视频到: {output_path}")
    
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_frames]
    
    try:
        imageio.mimsave(output_path, frames_rgb, fps=20, codec='libx264', quality=8)
        print(f"✓ 视频保存成功! ({len(all_frames)} 帧, {len(all_frames)/20:.1f}秒)")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 统计
    print("\n" + "="*70)
    print("推理测试完成!")
    print("="*70)
    print(f"Episodes数: {num_episodes}")
    print(f"成功率: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"平均步数: {np.mean(total_steps):.1f}")
    print(f"平均奖励: {np.mean(total_rewards):.3f}")
    print(f"视频文件: {output_path}")
    print(f"视频大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

