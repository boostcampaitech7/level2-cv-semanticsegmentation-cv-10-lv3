# %%
import os

# 원본 및 대상 경로 설정
source_dir = "./data/test_pseudo/DCM"
target_dir = "./data/new_test_pseudo/DCM"

# 원본 폴더 구조를 탐색하고 대상 경로에 생성
for root, dirs, files in os.walk(source_dir):
    relative_path = os.path.relpath(root, start=source_dir)
    target_path = os.path.join(target_dir, relative_path)

    os.makedirs(target_path, exist_ok=True)


print(f"폴더 구조가 {target_dir}에 생성 완료!")
