#!/usr/bin/env python3
"""
使用torch DCP API正确提取模型权重
"""
import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict

sys.path.insert(0, '/mnt/RL/yzy/RLinf')
from rlinf.models.embodiment.mlp_policy import MLPPolicy

# 设置环境
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

def main():
    print("="*70)
    print("使用DCP API提取训练好的模型")
    print("="*70)
    
    # 初始化分布式（单进程）
    print("\n1. 初始化分布式环境（单进程）...")
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    print("  ✓ 完成")
    
    # 创建模型
    print("\n2. 创建模型...")
    model = MLPPolicy(
        obs_dim=42,
        action_dim=8,
        hidden_dim=256,
        num_action_chunks=1,
        add_value_head=True
    )
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 准备state_dict结构
    print("\n3. 准备加载...")
    checkpoint_dir = "/mnt/RL/yzy/recordings/rlinf_logs/pickcube_mlp/checkpoints/global_step_2000/actor"
    
    state_dict = {'fsdp_checkpoint': {'model': model.state_dict()}}
    
    # 加载
    print(f"\n4. 从 {checkpoint_dir} 加载...")
    try:
        from torch.distributed.checkpoint import load
        load(state_dict, checkpoint_id=checkpoint_dir)
        
        # 提取模型权重
        model_state = state_dict['fsdp_checkpoint']['model']
        
        # 加载到模型
        model.load_state_dict(model_state)
        
        print("  ✓ 模型加载成功!")
        
        # 验证加载
        print("\n5. 验证模型...")
        print(f"  actor_mean.0.weight范围: [{model.actor_mean[0].weight.min().item():.4f}, {model.actor_mean[0].weight.max().item():.4f}]")
        print(f"  actor_logstd值: {model.actor_logstd.data.mean().item():.4f}")
        
        #保存为标准格式
        output_path = "/mnt/RL/yzy/recordings/trained_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'obs_dim': 42,
                'action_dim': 8,
                'hidden_dim': 256,
                'num_action_chunks': 1,
                'add_value_head': True,
            }
        }, output_path)
        
        print(f"\n✓ 模型已保存为标准PyTorch格式: {output_path}")
        
        # 清理
        dist.destroy_process_group()
        
        return True
        
    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        
        if dist.is_initialized():
            dist.destroy_process_group()
        
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*70)
        print("成功！现在可以使用trained_model.pth进行推理了")
        print("="*70)
    
    sys.exit(0 if success else 1)


