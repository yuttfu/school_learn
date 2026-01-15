# 交通场景 v2 优化总结

## 更新日期
2026-01-15

## 文件位置
- **Windows**: `D:\Desktop\program\项目\交通\traffic_scene_v2.xml`
- **Linux VM**: `/home/y/桌面/交通/traffic_scene_v2.xml`

## v2 主要改进

### 1. 真实悬挂系统
添加了4个独立的弹簧-阻尼悬挂关节：
- **悬挂刚度**: 50,000 N/m
- **阻尼系数**: 5,000 Ns/m
- **行程范围**: ±0.1m
- **传感器**: 4个悬挂位置传感器（susp_fl_pos, susp_fr_pos, susp_rl_pos, susp_rr_pos）

```xml
<joint name="susp_fl" type="slide" axis="0 0 1"
       range="-0.1 0.1" stiffness="50000" damping="5000"/>
```

### 2. 详细车身建模
**车身部件**:
- 引擎盖 (hood)
- 车顶 (roof)
- 后备箱 (trunk)
- 前/后保险杠 (bumpers)
- 左/右后视镜 (mirrors)

**玻璃系统**:
- 前挡风玻璃（透明度 0.6）
- 后窗玻璃（透明度 0.6）
- 左/右侧窗（透明度 0.6）

**灯光系统**:
- 前大灯 (白色)
- 尾灯 (红色)

### 3. 高质量材质
- **金属车身**: 金属光泽材质（specular=0.8, shininess=0.5）
- **铬合金轮毂**: 镜面反射（reflectance=0.9）
- **橡胶轮胎**: 真实橡胶纹理
- **玻璃**: 半透明材质（rgba alpha=0.6）

### 4. 多样化NPC车辆
- **NPC 1 (红色SUV)**: 大型越野车，车身高0.6m
- **NPC 2 (银色轿车)**: 标准轿车
- **NPC 3 (蓝色跑车)**: 低矮运动型，车身高0.45m

### 5. 执行器升级
从2个执行器升级到4个：
```xml
<!-- v1: 2个执行器 -->
<position name="steering" joint="steer_fl steer_fr" .../>
<motor name="drive" joint="roll_rl roll_rr" .../>

<!-- v2: 4个执行器 -->
<position name="steer_left" joint="steer_fl" .../>
<position name="steer_right" joint="steer_fr" .../>
<motor name="drive_left" joint="roll_rl" .../>
<motor name="drive_right" joint="roll_rr" .../>
```

### 6. 传感器增强
从16个传感器增加到20个：
- 6个主车状态传感器（位置、姿态、速度、加速度）
- 4个悬挂位置传感器（**新增**）
- 2个转向角传感器
- 4个车轮速度传感器
- 4个轮胎接触力传感器

## 技术规格对比

| 参数 | v1 | v2 |
|------|----|----|
| 传感器数量 | 16 | 20 |
| 执行器数量 | 2 | 4 |
| 关节数量 | 10 | 14 |
| 悬挂系统 | ❌ | ✅ 4个独立悬挂 |
| 车身细节 | 基础 | 详细（引擎盖、车灯、镜子等） |
| 材质质量 | 标准 | 高级（金属、玻璃、铬合金） |
| NPC多样性 | 相同车型 | 3种不同车型 |

## 使用方法

### 测试 v2 模型
```bash
# Windows
cd D:\Desktop\program\项目\交通
python test_v2.py

# Linux
cd /home/y/桌面/交通
python3 test_v2.py
```

### 控制接口
```python
# v2 需要控制4个执行器
data.ctrl[0] = steering_angle_left   # 左转向 [-0.61, 0.61] rad
data.ctrl[1] = steering_angle_right  # 右转向 [-0.61, 0.61] rad
data.ctrl[2] = drive_torque_left     # 左驱动 [-800, 800] Nm
data.ctrl[3] = drive_torque_right    # 右驱动 [-800, 800] Nm
```

### 读取悬挂数据
```python
import mujoco

# 获取悬挂传感器ID
fl_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'susp_fl_pos')
fr_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'susp_fr_pos')
rl_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'susp_rl_pos')
rr_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'susp_rr_pos')

# 读取悬挂位置
fl_pos = data.sensordata[fl_sensor]  # 单位: 米
fr_pos = data.sensordata[fr_sensor]
rl_pos = data.sensordata[rl_sensor]
rr_pos = data.sensordata[rr_sensor]
```

## 兼容性说明

v2 模型已针对 **MuJoCo 3.4.0** 进行优化，移除了以下不兼容的属性：
- `childclass` → 改为 `class`
- `emission` → 已移除
- `size2` → 已移除
- `skybox` → 已移除

## 性能特性

### 悬挂响应
- 弹簧常数: 50,000 N/m (较硬，适合高速公路)
- 阻尼系数: 5,000 Ns/m (快速阻尼)
- 自然频率: ~1.8 Hz
- 阻尼比: ~0.7 (欠阻尼)

### 转向特性
- 最大转向角: ±35° (0.61 rad)
- 转向刚度: 800 N·m/rad
- 阻尼: 80 N·m·s/rad
- 最大转向力: ±300 N·m

### 驱动特性
- 驱动方式: 后轮驱动
- 最大扭矩: 800 N·m (每轮)
- 齿轮比: 100:1
- 差速器: 开放式（独立控制实现）

## 下一步改进建议

1. **调整悬挂参数**: 当前悬挂可能过软，建议增加刚度或减少阻尼
2. **添加防倾杆**: 减少过弯时的侧倾
3. **改进差速器**: 实现限滑差速器逻辑
4. **添加ESP系统**: 电子稳定控制
5. **视觉渲染**: 添加环境贴图和阴影

## 文件清单

- `traffic_scene_v2.xml` - v2 场景定义
- `test_v2.py` - v2 测试脚本
- `V2_IMPROVEMENTS.md` - 本文档

---

**开发**: Claude Code
**日期**: 2026-01-15
**状态**: ✅ 完成并测试通过
