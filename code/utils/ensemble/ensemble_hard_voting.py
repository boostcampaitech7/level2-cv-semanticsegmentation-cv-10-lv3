import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def csv_ensemble(csv_paths, save_dir, threshold):
    def decode_rle_to_mask(rle, height, width):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(height * width, dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1

        return img.reshape(height, width)

    def encode_mask_to_rle(mask):
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    # csv의 기본 column
    csv_column = 8352

    csv_data = []
    for path in csv_paths:
        data = pd.read_csv(path)
        csv_data.append(data)

    file_num = len(csv_data)
    filename_and_class = []
    rles = []

    print(f"앙상블할 모델 수: {file_num}, threshold: {threshold}")

    for index in tqdm(range(csv_column)):
        model_rles = []
        for data in csv_data:
            if (type(data.iloc[index]['rle']) == float):
                model_rles.append(np.zeros((2048, 2048)))
                continue
            model_rles.append(decode_rle_to_mask(data.iloc[index]['rle'], 2048, 2048))

        image = np.zeros((2048, 2048))

        for model in model_rles:
            image += model

        # threshold 값으로 결정 (threshold의 값은 투표 수)
        # threshold로 설정된 값보다 크면 1, 작으면 0으로 변경
        image[image < threshold] = 0
        image[image >= threshold] = 1

        result_image = image

        rles.append(encode_mask_to_rle(result_image))
        filename_and_class.append(f"{csv_data[0].iloc[index]['class']}_{csv_data[0].iloc[index]['image_name']}")

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    # 기본 Dataframe 구조
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    # 최종 ensemble output 저장
    df.to_csv(save_dir, index=False)


csv_paths = []

for threshold in [3]:
    save_path = f"RESULT.csv"
    csv_ensemble(csv_paths, save_path, threshold)
