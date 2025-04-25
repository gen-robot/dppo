import coacd
import trimesh
import os
import numpy as np

model_dir = "/home/zhouzhiting/Projects/homebot/asset/real_assets"

for root, dirs, _ in os.walk(model_dir):
    for dir_name in dirs:  # 每个子文件夹代表一个对象类别
        sub_dir = os.path.join(root, dir_name)
        for file_name in os.listdir(sub_dir):
            if file_name.endswith(".glb"):
                input_file = os.path.join(sub_dir, file_name)
                # 使用 trimesh 加载 glb 文件
                mesh = trimesh.load(input_file, force="mesh")
                mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
                parts = coacd.run_coacd(mesh_coacd)
                
                # 将所有分解后的部分合并为一个整体 mesh
                all_vertices = []
                all_faces = []
                vertex_offset = 0

                for part in parts:
                    vertices = np.array(part[0], dtype=np.float32)
                    faces = np.array(part[1], dtype=np.int32)

                    all_vertices.append(vertices)
                    # 更新面的索引：
                    faces = faces + vertex_offset
                    all_faces.append(faces)
                    vertex_offset += len(vertices)

                combined_mesh = trimesh.Trimesh(
                    vertices=np.vstack(all_vertices),
                    faces=np.vstack(all_faces)
                )

                # 将结果保存到当前子文件夹下的 collision 子目录中
                collision_dir = os.path.join(sub_dir, "collision")
                os.makedirs(collision_dir, exist_ok=True)
                output_file = os.path.join(collision_dir, f"{os.path.splitext(file_name)[0]}_collision.obj")
                combined_mesh.export(output_file)