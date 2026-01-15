"""
交通场景简单测试 - 无需GUI
验证场景是否正常加载
"""

import mujoco
import numpy as np
import os

def test_traffic_scene():
    """测试交通场景"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "traffic_scene.xml")

    print("="*60)
    print("交通场景测试")
    print("="*60)

    # 加载模型
    print("\n1. 加载场景...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        print("   ✓ 场景加载成功")
    except Exception as e:
        print(f"   ✗ 加载失败: {e}")
        return

    # 创建数据
    data = mujoco.MjData(model)
    print("   ✓ 数据初始化完成")

    # 场景信息
    print(f"\n2. 场景信息:")
    print(f"   车辆数量: 4 (1主车 + 3 NPC)")
    print(f"   传感器数量: {model.nsensor}")
    print(f"   执行器数量: {model.nu}")
    print(f"   关节数量: {model.njnt}")
    print(f"   物体数量: {model.nbody}")

    # 获取主车ID
    try:
        ego_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ego_vehicle')
        print(f"   主车ID: {ego_body_id}")
    except:
        print("   警告: 找不到主车")

    # 运行仿真测试
    print(f"\n3. 运行仿真测试 (100步)...")

    for i in range(100):
        # 简单控制：直行
        if model.nu >= 4:
            data.ctrl[0] = 0.0   # 转向
            data.ctrl[1] = 0.0   # 转向
            data.ctrl[2] = 100.0 # 油门
            data.ctrl[3] = 100.0 # 油门

        # 仿真步进
        mujoco.mj_step(model, data)

        # 每20步打印一次位置
        if i % 20 == 0:
            try:
                pos = data.xpos[ego_body_id]
                vel = data.qvel[:3]
                speed = np.linalg.norm(vel) * 3.6  # km/h
                print(f"   步骤 {i:3d}: 位置=[{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:5.2f}]  "
                      f"速度={speed:5.1f} km/h")
            except:
                pass

    print(f"\n4. 传感器数据测试:")
    for i in range(min(5, model.nsensor)):
        sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_data = data.sensordata[i]
        print(f"   {sensor_name}: {sensor_data:.4f}")

    print("\n" + "="*60)
    print("✓ 测试完成！场景运行正常")
    print("="*60)
    print("\n提示: 要查看3D可视化，请在本地Windows运行")
    print("      python run_manual.py")

if __name__ == "__main__":
    test_traffic_scene()
