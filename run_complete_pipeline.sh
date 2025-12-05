#!/bin/bash
# 完整的训练到演示视频生成流程

echo "========================================"
echo "PickCube PPO 完整流程"
echo "========================================"
echo ""
echo "本脚本将执行以下步骤："
echo "  1. 训练PPO模型 (2000 epochs)"
echo "  2. 提取模型权重"
echo "  3. 生成演示视频"
echo ""
echo "预计总时长: ~4小时"
echo "========================================"
echo ""

# 激活环境
source /home/iclab/anaconda3/bin/activate sapien_yzy

# Step 1: 训练
echo "[Step 1/3] 开始训练..."
bash run_ppo_maniskill.sh

if [ $? -ne 0 ]; then
    echo "✗ 训练失败"
    exit 1
fi

echo ""
echo "✓ 训练完成"
echo ""

# Step 2: 提取模型
echo "[Step 2/3] 提取模型权重..."
python extract_with_dcp.py

if [ $? -ne 0 ]; then
    echo "✗ 模型提取失败"
    exit 1
fi

echo ""
echo "✓ 模型提取完成"
echo ""

# Step 3: 生成视频
echo "[Step 3/3] 生成演示视频..."
python demo_with_trained_model.py

if [ $? -ne 0 ]; then
    echo "✗ 视频生成失败"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ 完整流程执行成功！"
echo "========================================"
echo ""
echo "生成的文件："
echo "  - 训练日志: recordings/training_log.txt"
echo "  - 模型检查点: recordings/rlinf_logs/pickcube_mlp/checkpoints/"
echo "  - 提取的模型: recordings/trained_model.pth"
echo "  - 演示视频: recordings/1.mp4"
echo ""
echo "查看训练结果详情："
echo "  cat TRAINING_RESULTS_SUMMARY.md"
echo ""
echo "========================================"


