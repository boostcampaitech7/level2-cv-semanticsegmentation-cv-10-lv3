# %%
import numpy as np
import pandas as pd
import json
import os
from skimage import measure

# CSV 파일 경로와 저장할 JSON 파일 경로
csv_file_path = "./pseudo_csv/output_9742.csv"
json_output_dir = "./outputs_json_convert_9742"

os.makedirs(json_output_dir, exist_ok=True)

df = pd.read_csv(csv_file_path)

# %%


def rle_to_points(rle, height=2048, width=2048):
    """
    RLE 데이터를 디코딩하여 다각형 포인트로 변환하는 함수
    RLE 형식: "start1 length1 start2 length2 ..."
    """
    if pd.isna(rle):
        return []

    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    mask = mask.reshape((height, width))
    contours = measure.find_contours(mask, 0.5)

    points = []
    for contour in contours:
        contour_points = [[int(x), int(y)] for y, x in contour]
        points.extend(contour_points)

    return points


# %%
# 전체 JSON 생성 루프
for image_name, group in df.groupby('image_name'):
    json_file_name = os.path.splitext(image_name)[0] + ".json"
    json_file_path = os.path.join(json_output_dir, json_file_name)

    # JSON 구조 생성
    annotations = []
    for _, row in group.iterrows():
        points = rle_to_points(row['rle'])  # RLE 데이터를 points로 변환
        annotations.append({
            "id": f"{row['class']}",
            "type": "poly_seg",
            "attributes": {},
            "points": points,
            "label": row['class']
        })

    json_data = {
        "annotations": annotations,
        "filename": image_name,
    }

    # JSON 파일 저장
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

print(f"JSON 파일이 {json_output_dir} 폴더에 저장 완료!")
