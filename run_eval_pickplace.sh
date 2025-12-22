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
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# 创建必要的目录
mkdir -p /mnt/RL/yzy/recordings/eval_videos

echo "========================================"
echo "开始评估训练好的模型 - PickSingleYCB环境"
echo "========================================"
echo "Conda环境: sapien_yzy"
echo "GPU设备: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Episodes数: 10"
echo "检查点路径: /mnt/RL/yzy/recordings/rlinf_logs/pickplace_multishape_mlp/checkpoints/global_step_1200"
echo "视频保存路径: /mnt/RL/yzy/recordings/eval_videos"
echo "Python路径: $(which python)"
echo "========================================"

# 运行评估
cd ${EMBODIED_PATH}

python -u ${SRC_FILE} \
    --config-path /mnt/RL/yzy \
    --config-name maniskill_ppo_pickplace_eval \
    2>&1 | tee /mnt/RL/yzy/recordings/eval_log_pickplace.txt

EVAL_EXIT_CODE=$?

echo "========================================"
echo "评估完成!"
echo "日志保存在: /mnt/RL/yzy/recordings/eval_log_pickplace.txt"
echo "评估视频保存在: /mnt/RL/yzy/recordings/eval_videos"
echo "========================================"

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "评估成功完成！"
    echo "视频文件位置: /mnt/RL/yzy/recordings/eval_videos"
    ls -lh /mnt/RL/yzy/recordings/eval_videos/ 2>/dev/null | head -20
else
    echo "评估出现错误，请检查日志"
fi


