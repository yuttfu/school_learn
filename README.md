# MuJoCo 自动驾驶交通仿真系统

基于 MuJoCo 3.4.0 的高精度自动驾驶交通场景仿真平台，集成车辆动力学、传感器系统、控制策略（PID/MPC）和强化学习框架。

## 项目概述

本项目实现了一个完整的自动驾驶仿真环境，包含：
- **真实的车辆动力学模型**：4轮独立建模，包含转向、驱动、悬挂系统
- **多车道高速公路场景**：三车道道路，车道线标记，交通标志
- **多车交互**：主车 + 3辆NPC车辆
- **传感器系统**：激光雷达、相机、IMU、GPS、车轮编码器
- **控制策略**：PID、简化MPC
- **强化学习支持**：Gymnasium环境，支持SAC/TD3/PPO

## 场景特性

### 1. 道路系统
- ✅ 三车道高速公路（每车道宽3.6米）
- ✅ 100米长度道路
- ✅ 白色车道线标记（实线边界+虚线分隔）
- ✅ 沥青路面纹理
- ✅ 草地边缘

### 2. 车辆模型
**主车（Ego Vehicle）**：
- 车身尺寸：4.4m × 1.8m × 0.8m
- 质量：1500 kg（含惯性矩阵）
- 4个独立轮胎（直径0.7m）
- 前轮转向：±35度范围
- 后轮驱动：最大500 N·m扭矩
- 真实摩擦参数

**NPC车辆**：
- 3辆不同颜色的车辆（红、绿、黄）
- 分布在不同车道
- 简化动力学模型

### 3. 传感器系统
| 传感器类型 | 数量 | 说明 |
|-----------|------|------|
| 位置传感器 | 1 | 3D位置 |
| 姿态传感器 | 1 | 四元数姿态 |
| 速度传感器 | 2 | 线速度+角速度 |
| IMU | 2 | 加速度计+陀螺仪 |
| 车轮编码器 | 4 | 4轮转速 |
| 转向角编码器 | 2 | 前轮转向角 |
| 接触力传感器 | 4 | 轮胎接触力 |
| **激光雷达** | 1 | 360°扫描，360线 |
| **相机** | 1 | 640x480 RGB |

### 4. 物理参数（优化过的）
```xml
<!-- 求解器 -->
<option iterations="50" ls_iterations="20"/>

<!-- 车身惯性 -->
<inertial mass="1500" diaginertia="600 2000 2200"/>

<!-- 轮胎摩擦 -->
<geom friction="1.5 0.8 0.5" condim="3"/>

<!-- 关节阻尼 -->
<joint damping="0.5" frictionloss="0.1"/>
```

## 文件结构

```
交通/
├── traffic_scene.xml          # 交通场景定义
├── traffic_env.py             # Gymnasium 环境
├── train_rl.py                # RL 训练脚本
├── run_manual.py              # 手动控制脚本
├── requirements.txt           # 依赖列表
└── README.md                  # 本文档

生成的文件/
├── logs/                      # 训练日志
├── models/                    # 训练模型
└── videos/                    # 录制视频
```

## 安装

### 1. 基础安装（仅运行）

```bash
pip install mujoco>=3.4.0 numpy>=1.24.0 gymnasium>=0.29.0
```

### 2. 完整安装（包含RL训练）

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import mujoco; import gymnasium; print('安装成功!')"
```

## 使用方法

### 方式1：手动控制

```bash
python run_manual.py
```

**控制**：
- `↑` - 加速
- `↓` - 刹车
- `←` - 左转
- `→` - 右转
- `SPACE` - 重置场景
- `ESC` - 退出

### 方式2：PID控制演示

```bash
python train_rl.py --mode demo --n_episodes 3
```

演示PID控制器实现车道保持。

### 方式3：测试Gymnasium环境

```bash
python traffic_env.py
```

测试环境的功能：
- 观察空间和动作空间
- 传感器读取
- 奖励函数计算

### 方式4：强化学习训练

```bash
# 使用 SAC 算法训练
python train_rl.py --mode train --algorithm SAC --timesteps 1000000 --n_envs 4

# 使用 TD3 算法训练
python train_rl.py --mode train --algorithm TD3 --timesteps 1000000 --n_envs 4

# 使用 PPO 算法训练
python train_rl.py --mode train --algorithm PPO --timesteps 2000000 --n_envs 8
```

### 方式5：评估训练模型

```bash
python train_rl.py --mode evaluate --model_path models/SAC/best_model.zip --n_episodes 20
```

## 技术细节

### Gymnasium 环境

**观察空间**（9维）：
```python
[x, y, vx, vy, yaw, yaw_rate, lane_offset, heading_error, distance_to_goal]
```

**动作空间**（2维）：
```python
[steering_angle, throttle]  # 范围: [-1, 1]
```

**奖励函数**：
```python
reward = - lane_offset      # 车道保持
         - heading_error    # 航向保持
         - velocity_error   # 速度控制
         - action_penalty   # 平滑控制
         + progress_reward  # 前进奖励
         - collision_penalty # 碰撞惩罚
```

### 控制策略

#### PID控制器
```python
# 车道保持
steering = Kp * lane_offset + Ki * ∫(lane_offset)dt + Kd * d(lane_offset)/dt

# 速度控制
throttle = Kp * (target_speed - current_speed)
```

#### 简化MPC
- 预测时域：10步
- 线性化车辆模型
- 跟踪参考轨迹

### 激光雷达仿真

```python
# 360度扫描，每度一条射线
for angle in range(360):
    ray_direction = [cos(angle), sin(angle), 0]
    distance = mujoco.mj_ray(model, data, sensor_pos, ray_direction)
    ranges[angle] = distance
```

### 相机仿真

```python
renderer = mujoco.Renderer(model, height=480, width=640)
rgb_array = renderer.render()  # (480, 640, 3)
```

## 训练技巧

### 1. 超参数调优

**SAC（推荐用于连续控制）**：
```python
learning_rate = 3e-4
batch_size = 256
buffer_size = 200000
tau = 0.005
```

**TD3（稳定的off-policy）**：
```python
learning_rate = 3e-4
batch_size = 256
buffer_size = 200000
policy_delay = 2
```

**PPO（on-policy，易调试）**：
```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
```

### 2. 课程学习

```python
# 阶段1：简单车道保持（500k步）
target_velocity = 10.0 m/s
reward_weights = {'lane_keeping': -2.0}

# 阶段2：加速+保持（500k步）
target_velocity = 15.0 m/s
reward_weights = {'lane_keeping': -1.0, 'velocity': -0.1}

# 阶段3：完整任务（500k步）
target_velocity = 15.0 m/s
reward_weights = {'lane_keeping': -1.0, 'velocity': -0.1, 'collision': -100.0}
```

### 3. 并行训练

```bash
# 使用8个并行环境加速训练
python train_rl.py --n_envs 8 --timesteps 2000000
```

## 扩展功能

### 1. 添加更多车辆

在XML中添加：
```xml
<body name="npc_vehicle_4" pos="0 -3.6 0.3">
  <freejoint/>
  <inertial mass="1400" .../>
  <geom type="box" .../>
</body>
```

### 2. 实现变道功能

修改奖励函数：
```python
# 奖励变道到目标车道
target_lane = get_optimal_lane()
lane_change_reward = -abs(current_lane - target_lane)
```

### 3. 添加交通信号灯逻辑

```python
# 读取信号灯状态
light_state = get_traffic_light_state()

# 红灯停车
if light_state == 'red' and distance_to_light < 5.0:
    throttle = -1.0  # 刹车
```

### 4. 多智能体训练

```python
# 同时训练多辆车
class MultiAgentTrafficEnv(gym.Env):
    def __init__(self, n_agents=4):
        self.action_space = spaces.Tuple([
            spaces.Box(-1, 1, (2,)) for _ in range(n_agents)
        ])
```

### 5. 集成真实数据

```python
# 使用真实轨迹数据
import pandas as pd
trajectory_data = pd.read_csv('real_driving_data.csv')
reference_trajectory = trajectory_data[['x', 'y']].values
```

## 性能优化

### 1. 降低仿真复杂度
```xml
<!-- 减少求解器迭代 -->
<option iterations="30" ls_iterations="10"/>
```

### 2. 使用GPU加速（MJX）
```python
# 未来可以使用 MuJoCo XLA
import mujoco.mjx as mjx
```

### 3. 减少传感器频率
```python
# 每5步更新一次激光雷达
if step % 5 == 0:
    lidar_data = lidar.scan()
```

## 常见问题

### Q: 车辆翻倒怎么办？
A: 检查质心位置和惯性矩阵，降低车身高度或增加底盘质量。

### Q: 转向不灵敏？
A: 增加转向执行器的kp增益，或检查摩擦系数。

### Q: 训练不收敛？
A:
1. 降低学习率
2. 简化奖励函数（先只奖励车道保持）
3. 增加探索噪声
4. 使用课程学习

### Q: 如何录制视频？
A:
```python
import imageio
frames = []
for step in range(1000):
    rgb = env.render()  # mode='rgb_array'
    frames.append(rgb)
imageio.mimsave('video.mp4', frames, fps=30)
```

### Q: 如何添加雨雪天气？
A: 在XML中添加半透明粒子效果，调整摩擦系数模拟湿滑路面。

## 监控和可视化

### TensorBoard
```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 浏览器访问
http://localhost:6006
```

### 性能指标
- 平均回合奖励
- 车道偏移（米）
- 碰撞率
- 平均速度
- 成功到达终点率

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@software{mujoco_traffic_sim,
  title={MuJoCo Autonomous Driving Traffic Simulation},
  author={Claude Code + User},
  year={2026},
  url={https://github.com/your-repo}
}
```

## 参考资源

- [MuJoCo 官方文档](https://mujoco.readthedocs.io/)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [车辆动力学建模](https://www.researchgate.net/publication/vehicle-dynamics)

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0 (2026-01-15)
- ✅ 完整的交通场景
- ✅ 4轮车辆动力学
- ✅ 多传感器系统
- ✅ PID/MPC控制器
- ✅ Gymnasium环境
- ✅ 强化学习训练框架
- ✅ 激光雷达仿真
- ✅ 手动控制模式

---

**开发团队**: Claude Code + 用户协作
**最后更新**: 2026-01-15
**MuJoCo 版本**: 3.4.0
**Python 版本**: 3.8+
