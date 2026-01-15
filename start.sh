#!/bin/bash
# 交通仿真 - 快速启动脚本

echo "======================================"
echo "自动驾驶交通仿真 - 快速启动"
echo "======================================"
echo ""

# 检查依赖
echo "1. 检查依赖..."
python3 -c "import mujoco; import gymnasium; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ⚠ 缺少依赖，正在安装..."
    pip3 install --break-system-packages mujoco gymnasium numpy
else
    echo "   ✓ 所有依赖已安装"
fi

echo ""
echo "2. 可用的运行模式:"
echo "   [1] 手动控制（方向键）"
echo "   [2] PID控制演示"
echo "   [3] Gymnasium 环境测试"
echo "   [4] 强化学习演示（需要 stable-baselines3）"
echo ""
echo -n "请选择模式 (1-4): "
read choice

case $choice in
    1)
        echo ""
        echo "启动手动控制模式..."
        echo "控制: ↑加速 ↓刹车 ←左转 →右转 SPACE=重置"
        python3 run_manual.py
        ;;
    2)
        echo ""
        echo "启动PID控制演示..."
        python3 train_rl.py --mode demo --n_episodes 3
        ;;
    3)
        echo ""
        echo "测试 Gymnasium 环境..."
        python3 traffic_env.py
        ;;
    4)
        echo ""
        echo "检查 stable-baselines3..."
        python3 -c "import stable_baselines3" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "   安装 stable-baselines3..."
            pip3 install --break-system-packages stable-baselines3
        fi
        echo "启动 RL 随机策略演示..."
        python3 -c "from traffic_env import AutonomousDrivingEnv; import numpy as np; env = AutonomousDrivingEnv('traffic_scene.xml', render_mode='human'); obs, _ = env.reset(); [env.step(env.action_space.sample()) or env.render() for _ in range(500)]; env.close()"
        ;;
    *)
        echo "无效选择！"
        exit 1
        ;;
esac
