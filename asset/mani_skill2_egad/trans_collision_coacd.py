import coacd
import trimesh
import os
import numpy as np

# 遍历 model 文件夹，找到所有子文件夹下的 poisson 文件夹中的 textured.obj 文件
root_dir = "/home/zhouzhiting/Projects/homebot/asset/mani_skill2_egad"
# model_dir = os.path.join(root_dir, "egad_train_set")
model_dir = os.path.join(root_dir, "unique_files")


def trans(root, file):        
    # 构建完整的文件路径
    file_path = os.path.join(root, file)
    mesh = trimesh.load(file_path, force="mesh")
    # 将网格数据传递给 CoACD
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    # 运行 CoACD 算法
    parts = coacd.run_coacd(mesh)  # 返回一个凸包列表
    
    # 创建一个列表存储所有的顶点和面
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # 合并所有的 parts
    for part in parts:
        vertices = np.array(part[0], dtype=np.float32)
        faces = np.array(part[1], dtype=np.int32)
        
        # 添加顶点
        all_vertices.append(vertices)
        # 更新面的索引并添加
        faces = faces + vertex_offset
        all_faces.append(faces)
        
        # 更新顶点偏移量
        vertex_offset += len(vertices)
    
    # 创建合并后的mesh
    combined_mesh = trimesh.Trimesh(
        vertices=np.vstack(all_vertices),
        faces=np.vstack(all_faces)
    )

    return combined_mesh


for root, dirs, files in os.walk(model_dir):
    for file in files:
        # 检查文件是否是.obj文件
        if not file.endswith('.obj'):
            continue
        
        combined_mesh = trans(root, file)
        # 输出到 collision 文件夹
        collision_dir = os.path.join(root_dir, "egad_train_set_coacd")
        os.makedirs(collision_dir, exist_ok=True)
        output_file = os.path.join(collision_dir, file)
        combined_mesh.export(output_file)
