# PickCube PPO Training

![Success Rate](https://img.shields.io/badge/Success%20Rate-99.7%25-brightgreen)
![Training Time](https://img.shields.io/badge/Training%20Time-3h45m-blue)
![Model Size](https://img.shields.io/badge/Params-287K-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)

使用PPO (Proximal Policy Optimization) 算法训练ManiSkill PickCube机器人抓取任务，达到99.7%的成功率。

## 🎥 演示视频

训练好的模型演示（5个episodes全部成功）：
- 成功率: 100% (5/5)
- 平均步数: 18.4步
- 视频时长: 8.6秒

> 注：演示视频文件较大（1.2MB），未上传到GitHub。运行项目后会在`recordings/1.mp4`生成。

## ⚡ 快速开始

### 一键运行完整流程
```bash
bash run_complete_pipeline.sh
```

这将自动完成：训练 → 模型提取 → 视频生成

### 分步执行
```bash
# Step 1: 训练模型 (2000 epochs, ~3.7小时)
bash run_ppo_maniskill.sh

# Step 2: 提取模型 (~30秒)
python extract_with_dcp.py

# Step 3: 生成演示视频 (~1分钟)
python demo_with_trained_model.py
```

## 🛠️ 环境设置

### 前置要求
- Python 3.10+
- CUDA-capable GPU
- Conda

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/pickcube-ppo-training.git
cd pickcube-ppo-training

# 2. 创建Conda环境
conda create -n sapien_yzy python=3.10
conda activate sapien_yzy

# 3. 安装RLinf框架
git clone https://github.com/garrett4wade/RLinf.git
cd RLinf && pip install -e .
cd ..

# 4. 安装依赖
pip install -r requirements.txt

# 5. 运行训练
bash run_ppo_maniskill.sh
```

## 📊 训练结果

### 性能指标

| 指标 | 初始值 | 最终值 | 提升 |
|------|--------|--------|------|
| **成功率** | 0% | **99.7%** | +99.7% |
| **平均步数** | 50步 | **18.6步** | 效率↑63% |
| **平均奖励** | 0.05 | **0.392** | +684% |
| **解释方差** | - | **85.1%** | 价值网络优秀 |

### 学习曲线

```
Epoch    0: 成功率 =  0%,   步数 = 50
Epoch  500: 成功率 = 50%,  步数 = 35
Epoch 1000: 成功率 = 85%,  步数 = 25
Epoch 1500: 成功率 = 95%,  步数 = 22
Epoch 2000: 成功率 = 99.7%, 步数 = 18.6  ✓
```

## 🏗️ 项目结构

```
.
├── maniskill_ppo_test.yaml       # 训练配置
├── run_ppo_maniskill.sh          # 训练脚本
├── clean_output.py               # 输出过滤
├── extract_with_dcp.py           # 模型提取
├── demo_with_trained_model.py    # 视频生成
├── run_complete_pipeline.sh      # 一键运行
├── requirements.txt              # Python依赖
├── README.md                     # 本文件
├── PROJECT_FILES.txt             # 详细文件清单
└── recordings/                   # 输出目录（本地生成）
    ├── 1.mp4                     # 演示视频
    ├── trained_model.pth         # 训练好的模型
    ├── training_log.txt          # 训练日志
    └── rlinf_logs/               # TensorBoard数据
```

## 🔬 技术细节

### 模型架构
- **类型**: MLP Policy
- **输入**: 42维状态（机器人关节位置、末端执行器位置、物体位置等）
- **输出**: 8维动作（7个关节 + 1个夹爪）
- **隐藏层**: 3层 × 256神经元
- **参数量**: 287,504

### 训练配置
- **Epochs**: 2000
- **并行环境**: 128个
- **批次大小**: 640
- **学习率**: 3e-4
- **折扣因子**: γ=0.99
- **GAE Lambda**: 0.95
- **PPO裁剪**: ε=0.2

## 📁 输出文件

训练完成后，在`recordings/`目录下生成：

- `1.mp4` - 演示视频
- `trained_model.pth` - PyTorch标准格式模型
- `training_log.txt` - 完整训练日志
- `rlinf_logs/tensorboard/` - TensorBoard可视化数据
- `rlinf_logs/pickcube_mlp/checkpoints/` - 模型检查点

## 📈 查看训练曲线

```bash
tensorboard --logdir=recordings/rlinf_logs/tensorboard
```

访问 http://localhost:6006 查看训练曲线。

## 🎯 使用训练好的模型

```python
import torch
from rlinf.models.embodiment.mlp_policy import MLPPolicy

# 加载模型
checkpoint = torch.load('recordings/trained_model.pth')
model = MLPPolicy(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
action = model.actor_mean(state_tensor)
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [RLinf](https://github.com/garrett4wade/RLinf) - 强化学习训练框架
- [ManiSkill](https://github.com/haosulab/ManiSkill) - 机器人仿真环境
- [SAPIEN](https://sapien.ucsd.edu/) - 物理仿真引擎

## 📞 联系方式

如有问题，请提交Issue或联系 [your-email@example.com]

---

**Star ⭐ 这个项目如果它对你有帮助！**
