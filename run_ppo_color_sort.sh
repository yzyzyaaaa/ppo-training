#!/bin/bash

# 激活conda环境
source /home/iclab/anaconda3/bin/activate sapien_yzy

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1  # 使用GPU 1
export OMP_NUM_THREADS=16      # 限制OpenMP线程数
export MKL_NUM_THREADS=16      # 限制MKL线程数

# 设置MuJoCo和OpenGL相关环境变量
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# 设置日志级别为WARNING，减少无用输出
export LOG_LEVEL="ERROR"
export SAPIEN_LOG_LEVEL="off"
export PYTHONWARNINGS="ignore"
export RAY_LOGGING_LEVEL="ERROR"
export RAY_SILENCE_DEPRECATED_WARNINGS="1"

# 设置路径
export EMBODIED_PATH="/mnt/RL/yzy/RLinf/examples/embodiment"
export REPO_PATH="/mnt/RL/yzy/RLinf"
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# 创建必要的目录
mkdir -p /mnt/RL/yzy/recordings/rlinf_logs
mkdir -p /mnt/RL/yzy/recordings/rlinf_logs/checkpoints
mkdir -p /mnt/RL/yzy/recordings/rlinf_logs/video/train
mkdir -p /mnt/RL/yzy/recordings/eval_videos

echo "========================================"
echo "开始PPO训练 - 基于CLIP的颜色分类放置任务"
echo "========================================"
echo "Conda环境: sapien_yzy"
echo "GPU设备: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "训练轮数: 1200 epochs"
echo "环境数: 64"
echo "任务Prompt: Please place objects of different colors in separate areas. The specific rule is to put dark-colored items on the right side of the robotic arm and light-colored items on the left side."
echo "模型: CLIP-MLP Policy (视觉-语言对齐 + 强化学习)"
echo "日志路径: /mnt/RL/yzy/recordings/rlinf_logs"
echo "Python路径: $(which python)"
echo "========================================"

# 运行训练
cd ${EMBODIED_PATH}

python -u ${SRC_FILE} \
    --config-path /mnt/RL/yzy \
    --config-name maniskill_ppo_color_sort \
    2>&1 | python3 /mnt/RL/yzy/clean_output.py | tee /mnt/RL/yzy/recordings/training_log_color_sort.txt

TRAIN_EXIT_CODE=$?

echo "========================================"
echo "训练完成!"
echo "日志保存在: /mnt/RL/yzy/recordings/training_log_color_sort.txt"
echo "TensorBoard日志: /mnt/RL/yzy/recordings/rlinf_logs"
echo "========================================"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "训练成功完成！"
    echo "模型检查点保存在: /mnt/RL/yzy/recordings/rlinf_logs/checkpoints"
    echo "评估视频保存在: /mnt/RL/yzy/recordings/eval_videos"
else
    echo "训练出现错误，请检查日志"
fi

