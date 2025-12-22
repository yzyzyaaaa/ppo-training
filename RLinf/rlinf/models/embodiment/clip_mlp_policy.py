# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
try:
    import clip
except ImportError:
    raise ImportError(
        "CLIP is not installed. Please install it with: "
        "pip install git+https://github.com/openai/CLIP.git"
    )

from .modules.utils import layer_init
from .modules.value_head import ValueHead


class CLIPMLPPolicy(nn.Module):
    """
    基于CLIP视觉-语言对齐的MLP策略网络
    整合视觉观察、文本prompt和状态信息
    """
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        hidden_dim, 
        num_action_chunks, 
        add_value_head,
        clip_model_name="ViT-B/32",
        freeze_clip=True,
        use_state=True,
        vision_embed_dim=512,
        text_embed_dim=512,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.use_state = use_state
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        
        # 加载CLIP模型（先加载到CPU，避免初始化时的GPU内存冲突）
        print(f"Loading CLIP model: {clip_model_name}")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        
        # 将CLIP模型转换为float32以确保与FSDP和其他层的dtype一致
        self.clip_model = self.clip_model.float()
        print("Converted CLIP model to float32 for dtype consistency")
        
        # 注意：CLIP模型将在FSDP包装时自动移动到正确的GPU设备
        # 这里不立即移动到CUDA，避免在初始化阶段造成GPU内存冲突
        
        # 修复FSDP不支持的标量参数问题：将logit_scale从标量转换为1D张量
        if hasattr(self.clip_model, 'logit_scale'):
            logit_scale_value = self.clip_model.logit_scale.item()
            logit_scale_device = self.clip_model.logit_scale.device
            # 将logit_scale从标量Parameter转换为1D张量Parameter（FSDP要求）
            self.clip_model.logit_scale = nn.Parameter(torch.tensor([logit_scale_value], device=logit_scale_device, dtype=torch.float32))
            print(f"Fixed logit_scale shape: {self.clip_model.logit_scale.shape}, dtype: {self.clip_model.logit_scale.dtype}")
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # 投影层：将CLIP视觉和文本特征投影到统一空间
        self.vision_proj = nn.Sequential(
            layer_init(nn.Linear(vision_embed_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
        )
        self.text_proj = nn.Sequential(
            layer_init(nn.Linear(text_embed_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
        )
        
        # 特征融合层
        input_dim = hidden_dim * 2  # vision + text
        if use_state:
            input_dim += obs_dim  # 加上状态信息
        
        # Actor网络
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        # Value head
        self.value_head = (
            ValueHead(input_dim, hidden_sizes=(256, 256, 256), activation="tanh")
            if add_value_head
            else None
        )
        
        # 文本tokenizer（用于预处理prompt）
        self.text_tokenizer = clip.tokenize

    def encode_text(self, text_prompts):
        """使用CLIP编码文本prompt"""
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # Tokenize文本
        text_tokens = clip.tokenize(text_prompts, truncate=True).to(next(self.clip_model.parameters()).device)
        
        # 编码文本
        with torch.no_grad() if not self.clip_model.training else torch.enable_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # L2归一化
        
        return text_features

    def encode_image(self, images):
        """使用CLIP编码图像"""
        # images应该是[B, C, H, W]格式，值在[0, 1]范围或已归一化
        # CLIP期望输入为[0, 1]范围的float32
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        if images.max() > 1.0:
            images = images / 255.0
        
        # 确保在正确的设备上
        device = next(self.clip_model.parameters()).device
        if not images.is_cuda:
            images = images.to(device)
        
        # CLIP期望输入为[B, C, H, W]，H=W=224
        # 如果输入尺寸不对，需要resize
        if len(images.shape) == 4:
            if images.shape[-1] != 224 or images.shape[-2] != 224:
                images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        elif len(images.shape) == 3:
            # 单张图像，添加batch维度
            images = images.unsqueeze(0)
            if images.shape[-1] != 224 or images.shape[-2] != 224:
                images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 编码图像
        with torch.no_grad() if not self.clip_model.training else torch.enable_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # L2归一化
        
        return image_features

    def preprocess_obs(self, env_obs, text_prompt=None):
        """
        预处理观察值
        env_obs: 包含'images'和'states'的字典
        text_prompt: 文本提示（字符串或列表）
        """
        images = env_obs["images"]  # [B, C, H, W]
        states = env_obs["states"]  # [B, obs_dim]
        
        # 编码视觉和文本
        vision_features = self.encode_image(images)  # [B, vision_embed_dim]
        vision_features = self.vision_proj(vision_features)  # [B, hidden_dim]
        
        if text_prompt is not None:
            text_features = self.encode_text(text_prompt)  # [B, text_embed_dim]
            text_features = self.text_proj(text_features)  # [B, hidden_dim]
        else:
            # 如果没有提供文本，使用零向量
            batch_size = vision_features.shape[0]
            text_features = torch.zeros(batch_size, vision_features.shape[1], 
                                      device=vision_features.device)
        
        # 融合特征
        if self.use_state:
            # 拼接视觉、文本和状态特征
            states = states.to(vision_features.device)
            combined_features = torch.cat([vision_features, text_features, states], dim=-1)
        else:
            combined_features = torch.cat([vision_features, text_features], dim=-1)
        
        return combined_features

    def forward(
        self,
        data,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        obs = data["obs"]
        action = data["action"]
        text_prompt = data.get("text_prompt", None)
        
        # 预处理观察值（包含视觉、文本和状态）
        features = self.preprocess_obs(obs, text_prompt)
        
        action_mean = self.actor_mean(features)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        ret_dict = {}
        if compute_logprobs:
            logprobs = probs.log_prob(action)
            ret_dict["logprobs"] = logprobs
        if compute_entropy:
            entropy = probs.entropy()
            ret_dict["entropy"] = entropy
        if compute_values:
            values = self.value_head(features)
            ret_dict["values"] = values
        return ret_dict

    def predict_action_batch(
        self, env_obs, text_prompt=None, calulate_logprobs=True, calulate_values=True, **kwargs
    ):
        """批量预测动作"""
        features = self.preprocess_obs(env_obs, text_prompt)
        action_mean = self.actor_mean(features)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()

        chunk_actions = (
            action.reshape(-1, self.num_action_chunks, self.action_dim).cpu().numpy()
        )
        chunk_logprobs = probs.log_prob(action)

        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(features)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs = {"obs": env_obs, "action": action, "text_prompt": text_prompt}
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

