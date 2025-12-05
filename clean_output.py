#!/usr/bin/env python3
"""
清理训练输出，过滤掉WARNING、raylet等无用信息
"""
import sys
import re

# 要过滤掉的关键词
FILTER_KEYWORDS = [
    'WARNING',
    'raylet',
    'Generating Rollout Epochs',
    'file_system_monitor',
    'Discarding the following keys',
    'TORCH_NCCL_AVOID_RECORD_STREAMS',
    '[W1204',
    'is over 95% full',
    '[Gloo]',  # Gloo通信信息
    'Rank 0 is connected',  # Rank连接信息
    'Rank 1 is connected',
]

# 要保留的关键词（即使包含过滤关键词也要保留）
KEEP_KEYWORDS = [
    'Global Step:',
    '====',
    '开始',
    '训练',
    'GPU设备',
    'Conda环境',
]

for line in sys.stdin:
    # 检查是否需要保留这一行
    should_keep = False
    
    # 如果包含保留关键词，则保留
    if any(keyword in line for keyword in KEEP_KEYWORDS):
        should_keep = True
    # 如果不包含任何过滤关键词，也保留
    elif not any(keyword in line for keyword in FILTER_KEYWORDS):
        should_keep = True
    
    if should_keep:
        print(line, end='', flush=True)

