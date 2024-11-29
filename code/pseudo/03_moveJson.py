# %%
import os
import shutil

# JSON 파일 경로와 폴더 경로 설정
json_files_dir = "./outputs_json_convert_9742"  # JSON 파일이 저장된 경로
# level2-cv-semanticsegmentation-cv-10-lv3\code\data\test_pseudo\ouput_json
folders_dir = "./data/test_pseudo_new/output_json"  # 대상 폴더 경로

# JSON 파일과 폴더 목록 가져오기
json_files = sorted([f for f in os.listdir(json_files_dir) if f.endswith(".json")])
folders = sorted([d for d in os.listdir(folders_dir) if os.path.isdir(os.path.join(folders_dir, d))])

# JSON 파일을 2개씩 폴더에 배치
json_index = 0
for folder in folders:
    folder_path = os.path.join(folders_dir, folder)
    for _ in range(2):  # 각 폴더에 2개의 JSON 파일 넣기
        if json_index < len(json_files):
            json_file = json_files[json_index]
            source_path = os.path.join(json_files_dir, json_file)
            target_path = os.path.join(folder_path, json_file)

            # JSON 파일 복사
            shutil.copy(source_path, target_path)
            print(f"Copied {json_file} -> {folder}")

            json_index += 1

print("JSON 파일 분배가 완료되었습니다.")
