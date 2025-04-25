import coacd
import trimesh
import os
import numpy as np

# 遍历 model 文件夹，找到所有子文件夹下的 poisson 文件夹中的 textured.obj 文件
model_dir = "/home/zhouzhiting/Projects/homebot/asset/mani_skill2_ycb/models"

for root, dirs, files in os.walk(model_dir):
    for dir_name in dirs:
        poisson_dir = os.path.join(root, dir_name, "tsdf")
        textured_obj_path = os.path.join(poisson_dir, "textured.obj")
        if os.path.isfile(textured_obj_path):
            input_file = textured_obj_path
            # 加载 .obj 文件
            mesh = trimesh.load(input_file, force="mesh")
            mesh = coacd.Mesh(mesh.vertices, mesh.faces)
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
            
            # 输出到 collision 文件夹
            collision_dir = os.path.join(root, dir_name, "collision")
            os.makedirs(collision_dir, exist_ok=True)
            output_file = os.path.join(collision_dir, "collision.obj")
            combined_mesh.export(output_file)



