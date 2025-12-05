#!/usr/bin/env python3
"""
使用提取的训练模型生成演示视频
"""
import os
import sys
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
import imageio
import cv2

sys.path.insert(0, '/mnt/RL/yzy/RLinf')
from rlinf.models.embodiment.mlp_policy import MLPPolicy

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def add_text_to_frame(frame, text, position=(20, 40), color=(0, 255, 0), font_scale=1.2):
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

def main():
    print("="*70)
    print("使用训练好的模型生成演示视频")
    print("="*70)
    
    # 加载模型
    print("\n1. 加载训练好的模型...")
    model_path = "/mnt/RL/yzy/recordings/trained_model.pth"
    
    if not os.path.exists(model_path):
        print(f"  ✗ 模型文件不存在: {model_path}")
        print("  请先运行 extract_with_dcp.py 提取模型")
        return False
    
    checkpoint = torch.load(model_path, map_location='cuda:0')
    
    model = MLPPolicy(
        obs_dim=checkpoint['model_config']['obs_dim'],
        action_dim=checkpoint['model_config']['action_dim'],
        hidden_dim=checkpoint['model_config']['hidden_dim'],
        num_action_chunks=checkpoint['model_config']['num_action_chunks'],
        add_value_head=checkpoint['model_config']['add_value_head'],
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    
    print(f"  ✓ 模型加载成功 ({sum(p.numel() for p in model.parameters()):,} 参数)")
    
    # 创建环境
    print("\n2. 创建ManiSkill环境...")
    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
        max_episode_steps=50,
        sim_backend="gpu",
        sensor_configs={
            "shader_pack": "default",
            "width": 640,
            "height": 480,
        }
    )
    print("  ✓ 环境创建成功")
    
    # 录制视频
    print("\n3. 开始录制episodes...")
    print("-"*70)
    
    all_frames = []
    success_count = 0
    total_steps = []
    total_rewards = []
    
    num_episodes = 5
    
    with torch.no_grad():
        for episode in range(num_episodes):
            obs, info = env.reset(seed=episode+42)
            episode_frames = []
            episode_reward = 0.0
            episode_success = False
            steps = 0
            
            print(f"\nEpisode {episode + 1}/{num_episodes}:")
            
            for step in range(50):
                # 渲染
                frame = env.render()
                if frame is not None:
                    if hasattr(frame, 'cpu'):
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
                    if 'agent' in obs:
                        if 'qpos' in obs['agent']:
                            qpos = obs['agent']['qpos']
                        else:
                            qpos = list(obs['agent'].values())[0]
                    elif 'extra' in obs:
                        tcp_pose = obs['extra']['tcp_pose'] if 'tcp_pose' in obs['extra'] else []
                        obj_pose = obs['extra'].get('obj_pose', [])
                        goal_pose = obs['extra'].get('goal_pose', [])
                        qpos = obs.get('agent', {}).get('qpos', [])
                        
                        # 组合所有状态
                        state_list = []
                        for s in [qpos, tcp_pose, obj_pose, goal_pose]:
                            if hasattr(s, 'cpu'):
                                s = s.cpu().numpy()
                            if hasattr(s, 'flatten'):
                                s = s.flatten()
                            elif isinstance(s, (list, tuple)):
                                s = np.array(s).flatten()
                            state_list.append(s)
                        qpos = np.concatenate([s for s in state_list if len(s) > 0])
                    else:
                        qpos = obs
                else:
                    qpos = obs
                
                # 转换为numpy并展平
                if hasattr(qpos, 'cpu'):
                    state = qpos.cpu().numpy().flatten()
                elif hasattr(qpos, 'flatten'):
                    state = qpos.flatten()
                else:
                    state = np.array(qpos).flatten()
                
                # 确保维度正确 (42维)
                if len(state) < 42:
                    state = np.pad(state, (0, 42 - len(state)))
                elif len(state) > 42:
                    state = state[:42]
                
                # 转换为tensor并使用模型预测
                state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
                action_mean = model.actor_mean(state_tensor)
                action = action_mean.squeeze(0).cpu().numpy()
                
                # 裁剪到合法范围
                action = np.clip(action, -1.0, 1.0)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 提取奖励
                if hasattr(reward, 'item'):
                    reward_val = reward.item()
                elif hasattr(reward, '__iter__'):
                    reward_val = float(reward[0]) if len(reward) > 0 else 0
                else:
                    reward_val = float(reward)
                
                episode_reward += reward_val
                steps = step + 1
                
                # 检查是否完成
                done = terminated or truncated
                if hasattr(done, 'item'):
                    done = done.item()
                elif hasattr(done, '__iter__'):
                    done = done[0] if len(done) > 0 else False
                
                # 检查成功
                if hasattr(info, 'get'):
                    success_info = info.get('success', False)
                elif isinstance(info, dict):
                    success_info = info.get('success', False)
                else:
                    success_info = False
                
                if hasattr(success_info, 'item'):
                    episode_success = success_info.item()
                elif hasattr(success_info, '__iter__'):
                    episode_success = success_info[0] if len(success_info) > 0 else False
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
            print(f"  步数: {steps}")
            print(f"  奖励: {episode_reward:.3f}")
            print(f"  状态: {status}")
            
            # 添加文字到帧
            episode_text = f"Episode {episode + 1} - TRAINED MODEL"
            if episode_success:
                episode_text += " - SUCCESS!"
                text_color = (0, 255, 0)  # 绿色
            else:
                episode_text += " - INCOMPLETE"
                text_color = (0, 165, 255)  # 橙色
            
            for idx, frame in enumerate(episode_frames):
                frame_with_text = frame.copy()
                add_text_to_frame(frame_with_text, episode_text, (20, 40), text_color, 1.0)
                add_text_to_frame(frame_with_text, f"Step: {idx+1}/{len(episode_frames)}", 
                                (20, 80), (255, 255, 255), 0.8)
                add_text_to_frame(frame_with_text, f"Reward: {episode_reward:.2f}", 
                                (20, 120), (255, 200, 0), 0.8)
                all_frames.append(frame_with_text)
            
            # 分隔帧
            if episode < num_episodes - 1:
                black_frame = np.zeros_like(episode_frames[0])
                add_text_to_frame(black_frame, "--- Next Episode ---", (180, 240), (255, 255, 255), 1.5)
                all_frames.extend([black_frame] * 20)
    
    env.close()
    
    # 保存视频
    print("\n" + "-"*70)
    output_path = "/mnt/RL/yzy/recordings/1.mp4"
    print(f"保存视频到: {output_path}")
    
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_frames]
    
    try:
        imageio.mimsave(output_path, frames_rgb, fps=20, codec='libx264', quality=8)
        print("✓ 视频保存成功!")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return False
    
    # 统计
    print("\n" + "="*70)
    print("训练模型演示视频生成完成!")
    print("="*70)
    print(f"Episodes数: {num_episodes}")
    print(f"成功率: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"平均步数: {np.mean(total_steps):.1f}")
    print(f"平均奖励: {np.mean(total_rewards):.3f}")
    print(f"视频时长: {len(all_frames)/20:.1f}秒 (20 FPS)")
    print(f"视频文件: {output_path}")
    print("="*70)
    
    # 清理分布式
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


