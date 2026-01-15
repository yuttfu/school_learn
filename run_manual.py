"""
交通场景简单运行脚本 - 手动控制
"""

import mujoco
import mujoco.viewer
import numpy as np
import os


class ManualController:
    """手动控制器"""
    def __init__(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.steering_sensitivity = 0.02
        self.throttle_sensitivity = 10.0

    def handle_keyboard(self, key_pressed: dict):
        """处理键盘输入"""
        # 转向控制
        if 37 in key_pressed and key_pressed[37]:  # 左箭头
            self.steering = min(self.steering + self.steering_sensitivity, 0.61)
        if 39 in key_pressed and key_pressed[39]:  # 右箭头
            self.steering = max(self.steering - self.steering_sensitivity, -0.61)

        # 自动回正
        if 37 not in key_pressed and 39 not in key_pressed:
            if abs(self.steering) > 0.01:
                self.steering *= 0.9
            else:
                self.steering = 0

        # 油门/刹车控制
        if 38 in key_pressed and key_pressed[38]:  # 上箭头 - 加速
            self.throttle = min(self.throttle + self.throttle_sensitivity, 500)
        elif 40 in key_pressed and key_pressed[40]:  # 下箭头 - 刹车
            self.throttle = max(self.throttle - self.throttle_sensitivity, -500)
        else:
            # 自动减速
            self.throttle *= 0.95


def main():
    # 获取文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "traffic_scene.xml")

    # 检查文件
    if not os.path.exists(xml_path):
        print(f"错误：找不到场景文件 {xml_path}")
        return

    # 加载模型
    print("正在加载交通场景...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        model = mujoco.MjModel.from_xml_string(xml_content)

    data = mujoco.MjData(model)

    print("场景加载成功！")
    print(f"车辆数量: 4 (1主车 + 3 NPC)")
    print(f"传感器数量: {model.nsensor}")
    print(f"执行器数量: {model.nu}")

    print("\n" + "="*60)
    print("控制说明:")
    print("  ↑ - 加速")
    print("  ↓ - 刹车")
    print("  ← - 左转")
    print("  → - 右转")
    print("  SPACE - 重置场景")
    print("  ESC - 退出")
    print("="*60 + "\n")

    # 手动控制器
    controller = ManualController()
    key_pressed = {}

    def key_callback(keycode):
        key_pressed[keycode] = True

        # 空格 - 重置
        if keycode == 32:
            mujoco.mj_resetData(model, data)
            controller.steering = 0
            controller.throttle = 0
            print("场景已重置")

    # 启动仿真
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 设置相机跟随主车
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ego_vehicle')
        viewer.cam.distance = 15.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        step_count = 0

        while viewer.is_running():
            # 处理控制
            controller.handle_keyboard(key_pressed)
            key_pressed.clear()

            # 应用控制
            data.ctrl[0] = controller.steering  # 左前轮转向
            data.ctrl[1] = controller.steering  # 右前轮转向
            data.ctrl[2] = controller.throttle  # 左后轮驱动
            data.ctrl[3] = controller.throttle  # 右后轮驱动

            # 仿真步进
            mujoco.mj_step(model, data)
            viewer.sync()

            # 定期打印信息
            step_count += 1
            if step_count % 100 == 0:
                # 获取主车位置
                ego_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ego_vehicle')
                pos = data.xpos[ego_body_id]

                # 获取速度
                vel = data.qvel[model.body_jntadr[ego_body_id]:model.body_jntadr[ego_body_id]+3]
                speed = np.linalg.norm(vel) * 3.6  # 转换为 km/h

                print(f"位置: [{pos[0]:6.2f}, {pos[1]:6.2f}]  "
                      f"速度: {speed:5.1f} km/h  "
                      f"转向: {np.rad2deg(controller.steering):5.1f}°", end='\r')

    print("\n仿真结束")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
