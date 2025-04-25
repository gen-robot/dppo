import json
import trimesh
import os
import numpy as np
from pathlib import Path

def get_obj_info(obj_path, dir_name):
    mesh = trimesh.load(obj_path)

    bbox_min = mesh.bounds[0].tolist()
    bbox_max = mesh.bounds[1].tolist()
    
    # 计算尺寸
    size = mesh.bounds[1] - mesh.bounds[0]
    max_dim = np.argmax(size)
    along = "xyz"[max_dim]  # 根据最长边确定方向
    
    scale = 1.0  # 实际应用中需要根据具体需求计算
    
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
    info = {}
    
    # allowed_dirs = {"along"} # v0 
    allowed_dirs = {"column"} # v1
    # allowed_dirs = {"side", "tiny"} # v2, "box"

    for dir_name in os.listdir(model_dir):
        if dir_name in allowed_dirs:
            dir_path = os.path.join(model_dir, dir_name)
            for subdir_name in os.listdir(dir_path):
                poisson_dir = os.path.join(dir_path, subdir_name, "poisson")
                textured_obj_path = os.path.join(poisson_dir, "textured.obj")
                if os.path.isfile(textured_obj_path):
                    # model_id = f"{dir_name}/{subdir_name}"
                    model_id = f"{subdir_name}"
                    try:
                        info[model_id] = get_obj_info(textured_obj_path, dir_name)
                    except Exception as e:
                        print(f"processing {textured_obj_path} wrong: {e}")

    # 保存为JSON文件
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    model_dir = "./models"
    output_file = "info_pick_v1.json"
    create_json_info(model_dir, output_file)

## info_pick_v0.json: along
## info_pick_v1.json: column