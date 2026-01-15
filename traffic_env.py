"""
自动驾驶交通仿真环境
集成：传感器系统、控制策略（PID/MPC）、Gymnasium接口
基于 MuJoCo 2025 最佳实践
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import os


class LidarSimulator:
    """
    激光雷达仿真器 - 使用射线检测
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.n_rays = 360  # 扫描线数量
        self.max_range = 100.0  # 最大探测距离（米）
        self.fov = 360  # 视野角度

    def scan(self, sensor_pos: np.ndarray, sensor_rot: np.ndarray) -> np.ndarray:
        """
        执行激光雷达扫描

        Args:
            sensor_pos: 传感器位置 [x, y, z]
            sensor_rot: 传感器旋转矩阵 (3x3)

        Returns:
            ranges: 距离数组 (n_rays,)
        """
        ranges = np.full(self.n_rays, self.max_range)

        for i in range(self.n_rays):
            angle = (i / self.n_rays) * np.deg2rad(self.fov)

            # 射线方向（在传感器坐标系中）
            local_dir = np.array([np.cos(angle), np.sin(angle), 0])

            # 转换到世界坐标系
            world_dir = sensor_rot @ local_dir

            # 射线检测
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                self.model, self.data,
                sensor_pos, world_dir,
                None, 1, -1, geom_id
            )

            if distance >= 0 and distance < self.max_range:
                ranges[i] = distance

        return ranges


class CameraSimulator:
    """
    相机仿真器
    """

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480):
        self.model = model
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def capture(self, data: mujoco.MjData, camera_id: int = -1) -> np.ndarray:
        """
        捕获相机图像

        Args:
            data: MuJoCo数据
            camera_id: 相机ID（-1表示自由相机）

        Returns:
            rgb_array: RGB图像 (height, width, 3)
        """
        self.renderer.update_scene(data, camera=camera_id)
        return self.renderer.render()


class VehicleSensors:
    """
    车辆传感器系统
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._init_sensor_indices()

    def _init_sensor_indices(self):
        """初始化传感器索引"""
        self.sensor_map = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self.sensor_map[name] = i

    def get_position(self) -> np.ndarray:
        """获取车辆位置"""
        if 'ego_position' in self.sensor_map:
            idx = self.sensor_map['ego_position']
            return self.data.sensordata[idx:idx+3].copy()
        return np.zeros(3)

    def get_orientation(self) -> np.ndarray:
        """获取车辆姿态（四元数）"""
        if 'ego_orientation' in self.sensor_map:
            idx = self.sensor_map['ego_orientation']
            return self.data.sensordata[idx:idx+4].copy()
        return np.array([1, 0, 0, 0])

    def get_velocity(self) -> np.ndarray:
        """获取线速度"""
        if 'ego_linvel' in self.sensor_map:
            idx = self.sensor_map['ego_linvel']
            return self.data.sensordata[idx:idx+3].copy()
        return np.zeros(3)

    def get_angular_velocity(self) -> np.ndarray:
        """获取角速度"""
        if 'ego_angvel' in self.sensor_map:
            idx = self.sensor_map['ego_angvel']
            return self.data.sensordata[idx:idx+3].copy()
        return np.zeros(3)

    def get_acceleration(self) -> np.ndarray:
        """获取加速度"""
        if 'ego_accel' in self.sensor_map:
            idx = self.sensor_map['ego_accel']
            return self.data.sensordata[idx:idx+3].copy()
        return np.zeros(3)

    def get_wheel_speeds(self) -> Dict[str, float]:
        """获取车轮转速"""
        speeds = {}
        for wheel in ['wheel_fl_speed', 'wheel_fr_speed', 'wheel_rl_speed', 'wheel_rr_speed']:
            if wheel in self.sensor_map:
                idx = self.sensor_map[wheel]
                speeds[wheel] = self.data.sensordata[idx]
        return speeds

    def get_steering_angles(self) -> Dict[str, float]:
        """获取转向角"""
        angles = {}
        for steer in ['steer_fl_angle', 'steer_fr_angle']:
            if steer in self.sensor_map:
                idx = self.sensor_map[steer]
                angles[steer] = self.data.sensordata[idx]
        return angles

    def get_tire_contacts(self) -> Dict[str, float]:
        """获取轮胎接触力"""
        contacts = {}
        for contact in ['contact_fl', 'contact_fr', 'contact_rl', 'contact_rr']:
            if contact in self.sensor_map:
                idx = self.sensor_map[contact]
                contacts[contact] = self.data.sensordata[idx]
        return contacts


class PIDController:
    """
    PID控制器 - 用于车道保持
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        """
        计算控制输出

        Args:
            error: 误差
            dt: 时间步长

        Returns:
            control: 控制输出
        """
        # 比例项
        p_term = self.kp * error

        # 积分项
        self.integral += error * dt
        i_term = self.ki * self.integral

        # 微分项
        if dt > 0:
            d_term = self.kd * (error - self.prev_error) / dt
        else:
            d_term = 0.0

        self.prev_error = error

        return p_term + i_term + d_term

    def reset(self):
        """重置控制器"""
        self.integral = 0.0
        self.prev_error = 0.0


class SimpleMPCController:
    """
    简化的模型预测控制器
    使用线性化模型进行轨迹跟踪
    """

    def __init__(self, horizon: int = 10, dt: float = 0.1):
        self.horizon = horizon
        self.dt = dt

    def compute(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
        current_velocity: float
    ) -> Tuple[float, float]:
        """
        计算MPC控制输出（简化版）

        Args:
            current_state: [x, y, yaw, v]
            reference_trajectory: 参考轨迹 [[x, y], ...]
            current_velocity: 当前速度

        Returns:
            steering_angle: 转向角
            throttle: 油门/刹车
        """
        # 简化：只计算横向误差和航向误差
        current_pos = current_state[:2]
        current_yaw = current_state[2]

        # 找最近的参考点
        if len(reference_trajectory) > 0:
            dists = np.linalg.norm(reference_trajectory - current_pos, axis=1)
            nearest_idx = np.argmin(dists)

            # 横向误差
            lateral_error = dists[nearest_idx]

            # 预测未来位置
            lookahead_idx = min(nearest_idx + 5, len(reference_trajectory) - 1)
            target_pos = reference_trajectory[lookahead_idx]

            # 计算目标航向
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            target_yaw = np.arctan2(dy, dx)

            # 航向误差
            yaw_error = self._normalize_angle(target_yaw - current_yaw)

            # 简化控制律
            steering_angle = -0.5 * lateral_error + 1.0 * yaw_error
            steering_angle = np.clip(steering_angle, -0.5, 0.5)

            # 速度控制
            target_velocity = 10.0  # m/s
            velocity_error = target_velocity - current_velocity
            throttle = 0.3 * velocity_error
            throttle = np.clip(throttle, -100, 500)

            return steering_angle, throttle
        else:
            return 0.0, 0.0

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class AutonomousDrivingEnv(gym.Env):
    """
    自动驾驶Gymnasium环境
    任务：车道保持、避障
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(
        self,
        xml_path: str,
        max_episode_steps: int = 1000,
        target_velocity: float = 15.0,  # m/s (约54 km/h)
        render_mode: Optional[str] = None
    ):
        super().__init__()

        # 加载模型
        if not os.path.exists(xml_path):
            xml_path = xml_path.replace('\\', '/')

        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        except:
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            self.model = mujoco.MjModel.from_xml_string(xml_content)

        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # 初始化传感器和控制器
        self.sensors = VehicleSensors(self.model, self.data)
        self.lidar = LidarSimulator(self.model, self.data)

        if render_mode == 'rgb_array':
            self.camera = CameraSimulator(self.model)

        # 定义动作空间：[转向角, 油门/刹车]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # 定义观察空间
        # [x, y, vx, vy, yaw, yaw_rate, lane_offset, heading_error, distance_to_goal]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self.target_velocity = target_velocity
        self.current_step = 0
        self.viewer = None

        # 奖励权重
        self.reward_weights = {
            'lane_keeping': -1.0,
            'heading': -0.5,
            'velocity': -0.1,
            'action_smooth': -0.01,
            'collision': -100.0,
            'progress': 1.0
        }

        # 目标车道中心线（中间车道）
        self.target_lane_y = 0.0

    def _get_obs(self) -> np.ndarray:
        """获取观察"""
        pos = self.sensors.get_position()
        vel = self.sensors.get_velocity()
        quat = self.sensors.get_orientation()

        # 从四元数计算偏航角
        yaw = self._quat_to_yaw(quat)

        # 计算角速度
        ang_vel = self.sensors.get_angular_velocity()
        yaw_rate = ang_vel[2]

        # 车道偏移
        lane_offset = pos[1] - self.target_lane_y

        # 航向误差（应该沿x轴行驶）
        target_heading = 0.0
        heading_error = self._normalize_angle(target_heading - yaw)

        # 到终点的距离
        goal_position = 40.0  # 终点x坐标
        distance_to_goal = goal_position - pos[0]

        obs = np.array([
            pos[0], pos[1],  # 位置
            vel[0], vel[1],  # 速度
            yaw, yaw_rate,   # 姿态
            lane_offset,     # 车道偏移
            heading_error,   # 航向误差
            distance_to_goal # 到终点距离
        ], dtype=np.float32)

        # 添加传感器噪声
        noise = np.random.normal(0, 0.01, obs.shape)
        obs += noise

        return obs

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """计算奖励"""
        pos = self.sensors.get_position()
        vel = self.sensors.get_velocity()

        # 车道保持奖励
        lane_offset = abs(pos[1] - self.target_lane_y)
        lane_reward = self.reward_weights['lane_keeping'] * lane_offset

        # 航向奖励
        quat = self.sensors.get_orientation()
        yaw = self._quat_to_yaw(quat)
        heading_error = abs(self._normalize_angle(0.0 - yaw))
        heading_reward = self.reward_weights['heading'] * heading_error

        # 速度奖励
        current_speed = np.linalg.norm(vel[:2])
        velocity_error = abs(current_speed - self.target_velocity)
        velocity_reward = self.reward_weights['velocity'] * velocity_error

        # 动作平滑性（避免抖动）
        action_penalty = self.reward_weights['action_smooth'] * np.sum(np.square(self.data.ctrl))

        # 前进奖励
        progress_reward = self.reward_weights['progress'] * vel[0]

        # 碰撞检测
        collision = self._check_collision()
        collision_penalty = self.reward_weights['collision'] if collision else 0

        # 离开道路检测
        off_road = abs(pos[1]) > 5.5
        off_road_penalty = -50.0 if off_road else 0

        # 总奖励
        total_reward = (lane_reward + heading_reward + velocity_reward +
                       action_penalty + progress_reward + collision_penalty +
                       off_road_penalty)

        # 终止条件
        terminated = collision or off_road or pos[0] > 40.0

        info = {
            'lane_offset': lane_offset,
            'heading_error': heading_error,
            'speed': current_speed,
            'collision': collision,
            'off_road': off_road,
            'position': pos.copy()
        }

        return total_reward, terminated, info

    def _check_collision(self) -> bool:
        """检测碰撞"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # 检查是否与其他车辆或障碍物碰撞
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            # 如果ego车辆与NPC车辆碰撞
            if geom1_name and geom2_name:
                if ('ego' in geom1_name and 'npc' in geom2_name) or \
                   ('npc' in geom1_name and 'ego' in geom2_name):
                    return True

        return False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        # 重置仿真
        mujoco.mj_resetData(self.model, self.data)

        # 随机化初始位置（在车道内）
        if options and options.get('randomize_init', True):
            # 随机选择车道
            lane_choices = [-3.6, 0.0, 3.6]  # 左、中、右车道
            init_lane = np.random.choice(lane_choices)

            # 设置主车初始位置
            ego_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ego_vehicle')
            self.data.qpos[self.model.body_jntadr[ego_body_id]:self.model.body_jntadr[ego_body_id]+3] = \
                [-20 + np.random.uniform(-2, 2), init_lane + np.random.uniform(-0.5, 0.5), 0.3]

        # 前向仿真几步稳定
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.current_step = 0

        obs = self._get_obs()
        info = {'sensors': {'position': self.sensors.get_position()}}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 应用动作
        action = np.clip(action, -1, 1)

        # 转向（弧度）
        steering_angle = action[0] * 0.61  # ±35度
        self.data.ctrl[0] = steering_angle  # 左前轮
        self.data.ctrl[1] = steering_angle  # 右前轮

        # 油门/刹车
        throttle = action[1] * 500  # ±500 N·m
        self.data.ctrl[2] = throttle  # 左后轮
        self.data.ctrl[3] = throttle  # 右后轮

        # 执行仿真步
        for _ in range(4):  # 每个gym步骤执行多个物理步骤
            mujoco.mj_step(self.model, self.data)

        # 获取观察和奖励
        obs = self._get_obs()
        reward, terminated, info = self._compute_reward()

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        info['step'] = self.current_step

        return obs, reward, terminated, truncated, info

    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # 设置相机视角（跟随车辆）
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, 'ego_vehicle'
                )
                self.viewer.cam.distance = 15.0
                self.viewer.cam.elevation = -20
                self.viewer.cam.azimuth = 90
            self.viewer.sync()
        elif self.render_mode == 'rgb_array':
            return self.camera.capture(self.data)

    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        """四元数转偏航角"""
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def test_environment():
    """测试环境"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "traffic_scene.xml")

    print("创建自动驾驶环境...")
    env = AutonomousDrivingEnv(
        xml_path=xml_path,
        max_episode_steps=1000,
        render_mode='human'
    )

    print("环境信息:")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  传感器数量: {env.model.nsensor}")

    print("\n开始测试...")

    # 使用PID控制器测试
    lane_controller = PIDController(kp=0.5, ki=0.05, kd=0.2)
    heading_controller = PIDController(kp=1.0, ki=0.1, kd=0.5)

    for episode in range(3):
        obs, info = env.reset()
        total_reward = 0
        lane_controller.reset()
        heading_controller.reset()

        print(f"\n=== Episode {episode + 1} ===")

        for step in range(1000):
            # PID控制
            lane_offset = obs[6]
            heading_error = obs[7]

            steering = lane_controller.compute(lane_offset, 0.02) + \
                      heading_controller.compute(heading_error, 0.02)
            steering = np.clip(steering, -1, 1)

            # 简单速度控制
            current_speed = np.linalg.norm(obs[2:4])
            throttle = 0.3 * (15.0 - current_speed)
            throttle = np.clip(throttle, -1, 1)

            action = np.array([steering, throttle])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env.render()

            if terminated or truncated:
                print(f"回合结束 - 步数: {step + 1}, 总奖励: {total_reward:.2f}")
                print(f"  最终位置: {info['position']}")
                print(f"  车道偏移: {info['lane_offset']:.3f}m")
                print(f"  碰撞: {info['collision']}")
                break

    env.close()
    print("\n测试完成!")


if __name__ == "__main__":
    test_environment()
