#!/bin/bash
# 监控资源下载进度

echo "========================================"
echo "监控bridge_v2_real2sim资源下载进度"
echo "========================================"

ASSET_DIR="$HOME/.maniskill/data/tasks/bridge_v2_real2sim_dataset"
REQUIRED_FILE="$ASSET_DIR/custom/models/bridge_carrot_generated_modified/textured.glb"

echo ""
echo "检查资源目录..."
if [ -d "$ASSET_DIR" ]; then
    echo "✓ 资源目录存在: $ASSET_DIR"
    
    # 计算目录大小
    SIZE=$(du -sh "$ASSET_DIR" 2>/dev/null | cut -f1)
    echo "✓ 目录大小: $SIZE"
    
    # 检查models目录
    if [ -d "$ASSET_DIR/custom/models" ]; then
        MODEL_COUNT=$(find "$ASSET_DIR/custom/models" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "✓ 找到 $MODEL_COUNT 个模型目录"
    fi
    
    # 检查关键文件
    if [ -f "$REQUIRED_FILE" ]; then
        echo ""
        echo "========================================"
        echo "✓ 资源下载完成！关键文件存在"
        echo "========================================"
        echo "文件: $REQUIRED_FILE"
        exit 0
    else
        echo ""
        echo "⚠ 关键文件不存在: $REQUIRED_FILE"
        echo ""
        echo "下载可能正在进行中..."
        echo "查看下载日志: tail -f /mnt/RL/yzy/download_output.log"
    fi
else
    echo "⚠ 资源目录不存在: $ASSET_DIR"
    echo ""
    echo "下载可能正在进行中..."
    echo "查看下载日志: tail -f /mnt/RL/yzy/download_output.log"
fi

echo ""
echo "========================================"
echo "下载状态检查完成"
echo "========================================"
