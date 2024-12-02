# %%
import os
import shutil

# JSON 파일 경로와 폴더 경로 설정
json_files_dir = "./outputs_json_convert_9742"
folders_dir = "./data/test_pseudo_new/output_json"

json_files = sorted([f for f in os.listdir(json_files_dir) if f.endswith(".json")])
folders = sorted([d for d in os.listdir(folders_dir) if os.path.isdir(os.path.join(folders_dir, d))])

json_index = 0
for folder in folders:
    folder_path = os.path.join(folders_dir, folder)
    for _ in range(2):
        if json_index < len(json_files):
            json_file = json_files[json_index]
            source_path = os.path.join(json_files_dir, json_file)
            target_path = os.path.join(folder_path, json_file)

            shutil.copy(source_path, target_path)
            print(f"Copied {json_file} -> {folder}")

            json_index += 1

print("JSON 파일 분배 완료!")
