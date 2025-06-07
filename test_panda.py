import sapien.core as sapien
import numpy as np
import os
from env.robot import load_robot_panda

# 初始化SAPIEN引擎
engine = sapien.Engine()
renderer = sapien.SapienRenderer(offscreen_only=True)
engine.set_renderer(renderer)

# 创建场景
scene = engine.create_scene()
scene.set_timestep(1/240)

# 设置相机
camera = scene.add_camera(
    name="camera",
    width=128,
    height=128,
    fovy=0.6911112070083618,  # 只使用fovy参数
    near=0.1,
    far=1000.0,
)
camera.set_pose(sapien.Pose([0, 0, 2], [1, 0, 0, 0]))

# 添加光源
scene.set_ambient_light([0.5, 0.5, 0.5])
directional_light = scene.add_directional_light(
    direction=[0, 0, -1],
    color=[1, 1, 1],
    shadow=True,
)

# 检查URDF文件是否存在
urdf_path = '/home/susichang/sichang/dppo/asset/franka_panda/panda.urdf'
print(f"Checking URDF file: {urdf_path}")
if os.path.exists(urdf_path):
    print("URDF file exists!")
else:
    print("URDF file does not exist!")
    exit(1)

try:
    # 加载机器人
    print("Attempting to load Panda robot...")
    robot, finger_link_idxs = load_robot_panda(scene)
    print("Panda robot loaded successfully!")
    print(f"Number of finger links: {len(finger_link_idxs)}")
    
    # 打印机器人信息
    print("\nRobot information:")
    print(f"Robot name: {robot.get_name()}")
    print(f"Number of links: {len(robot.get_links())}")
    print(f"Number of joints: {len(robot.get_active_joints())}")
    
    # 渲染一帧
    scene.step()
    scene.update_render()
    camera.take_picture()
    
    print("\nScene rendered successfully!")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    # 清理资源
    scene = None
    renderer = None
    engine = None 