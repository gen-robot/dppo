import os
import shutil
import json

# Define the paths to the directories
egad_train_set = '/home/zhouzhiting/Projects/homebot/asset/mani_skill2_egad/egad_train_set'
egad_train_set_coacd = '/home/zhouzhiting/Projects/homebot/asset/mani_skill2_egad/egad_train_set_coacd'
unique_dir = '/home/zhouzhiting/Projects/homebot/asset/mani_skill2_egad/unique_files'

files_in_egad_train_set = set(os.listdir(egad_train_set))
files_in_egad_train_set_coacd = set(os.listdir(egad_train_set_coacd))

num_files_in_egad_train_set = len(os.listdir(egad_train_set))
num_files_in_egad_train_set_coacd = len(os.listdir(egad_train_set_coacd))
print(f"egad_train_set: {num_files_in_egad_train_set}")
print(f"egad_train_set_coacd: {num_files_in_egad_train_set_coacd}")

# Count the number of entries in egad_train_set_coacd that are in info_pick_train_v0.json
json_path_v0 = '/home/zhouzhiting/Projects/homebot/asset/mani_skill2_egad/info_pick_train_v0.json' # 1600
json_path_v1 = '/home/zhouzhiting/Projects/homebot/asset/mani_skill2_egad/info_pick_train_v1.json' # 1331
# raw_train 2281

with open(json_path_v0, 'r') as f:
    data_v0 = json.load(f)
    num_entries_v0 = len(data_v0)
keys_in_json_v0 = set(data_v0.keys())
with open(json_path_v1, 'r') as f:
    data_v1 = json.load(f)
    num_entries_v1 = len(data_v1)
keys_in_json_v1 = set(data_v1.keys())
# print(f"Number of entries in json (v0): {num_entries_v0}")
# print(f"Number of entries in json (v1): {num_entries_v1}")


# # (1) Find the intersection of the keys and the files
files_in_egad_train_set_coacd_no_ext = {os.path.splitext(file)[0] for file in files_in_egad_train_set_coacd}
matching_entries_v0 = keys_in_json_v0 & files_in_egad_train_set_coacd_no_ext
matching_entries_v1 = keys_in_json_v1 & files_in_egad_train_set_coacd_no_ext
print(f"matching process(v0): {len(matching_entries_v0)} / {num_entries_v0}")
print(f"matching process(v1): {len(matching_entries_v1)} / {num_entries_v1}")

# #  (2) Move the unique files to the new output directory
# Ensure the unique directory exists
if not os.path.exists(unique_dir):
    os.makedirs(unique_dir)

# Find the .obj files that are in egad_train_set_coacd but not in egad_train_set
diff_files = files_in_egad_train_set - files_in_egad_train_set_coacd
print(f"Number of unique files: {len(diff_files)}")  # 342 

# Move the unique .obj files to the unique directory and delete the rest
# for file in diff_files:
#     src = os.path.join(egad_train_set, file)
#     dst = os.path.join(unique_dir, file)
#     shutil.copy(src, dst)
