import json
import trimesh
import os
import numpy as np
from pathlib import Path

def get_obj_info(obj_path, dir_name):
    mesh = trimesh.load(obj_path)

    bbox_min = mesh.bounds[0].tolist()
    bbox_max = mesh.bounds[1].tolist()

    size = mesh.bounds[1] - mesh.bounds[0]
    max_dim = np.argmax(size)
    along = "xyz"[max_dim] 
    
    scale = 1.0  
    
    return {
        "bbox": {
            "min": bbox_min,
            "max": bbox_max
        },
        "scales": [float(scale)],
        "along": along,
        "allow_dir": dir_name
    }

def create_json_info(model_dir, output_file):
    # 如果输出文件已存在，则读取已有信息，否则新建字典
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            info = json.load(f)
    else:
        info = {}
    
    # allowed_dirs = {"along"} # v0 , along
    # allowed_dirs = {"column"}  # v1, column
    allowed_dirs = {"side"}    # v2, side

    for dir_name in os.listdir(model_dir):
        if dir_name in allowed_dirs:
            dir_path = os.path.join(model_dir, dir_name)
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".glb"):
                        glb_path = os.path.join(root, file)
                        model_id = os.path.splitext(file)[0]
                        # 如果该物体已在 info 中，则不更新
                        if model_id in info:
                            continue
                        try:
                            info[model_id] = get_obj_info(glb_path, dir_name)
                        except Exception as e:
                            print(f"Processing {glb_path} failed: {e}")
    
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2)
    
if __name__ == "__main__":
    model_dir = "./"
    output_file = "info_pick_v2.json"
    create_json_info(model_dir, output_file)

## info_pick_v0.json: along
## info_pick_v1.json: column
## info_pick_v2.json: side