import os

# 모델 실험 결과 저장된 폴더 경로
input_path = os.path.join(os.getcwd(), "outputs/ensemble_input")
output_path = os.path.join(os.getcwd(), "outputs/ensemble_output")
crop_file_path = os.path.join(os.getcwd(), "crop.csv")  # crop.csv 경로 설정

# ensemble_output 폴더 없으면 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"{output_path} 폴더가 생성")

# 클래스 이름과 인덱스 매핑
all_classes = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5', 'finger-6', 'finger-7', 'finger-8', 'finger-9',
    'finger-10', 'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15', 'finger-16', 'finger-17',
    'finger-18', 'finger-19', 'Trapezium', 'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum',
    'Pisiform', 'Radius', 'Ulna'
]
selected_classes = ['Trapezium', 'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum', 'Pisiform']
selected_class_indices = [all_classes.index(cls) for cls in selected_classes]

# Output File 생성
output_file_path = os.path.join(output_path, f"selsect_ensemble_from_{'_'.join(os.listdir(input_path))}.csv")
output_file = open(output_file_path, "w")

# Input File 로드
print("Load Input CSV File...")
inputs_files_data = []
data_length = None
header_line = None
for exp in os.listdir(input_path):
    exp_path = os.path.join(input_path, exp)
    input_file_name = [f for f in os.listdir(exp_path) if f.endswith(".csv")][0]

    with open(os.path.join(exp_path, input_file_name), "r") as file:
        data = file.readlines()
        inputs_files_data.append(data)
        if data_length is None:
            data_length = len(data)
        if header_line is None:
            header_line = data[0]

# crop.csv 로드
print("Load Crop CSV File...")
with open(crop_file_path, "r") as crop_file:
    crop_data = crop_file.readlines()

# crop.csv와 앙상블 파일의 길이 검증
assert len(crop_data) == data_length, "crop.csv와 입력 데이터의 길이가 일치하지 않습니다!"

# Ensemble 및 crop.csv 값으로 대체
print("Ensemble...")
for i in range(data_length):
    if i == 0:
        # Header 작성
        output_file.write(header_line)
    else:
        
        class_index = (i % len(all_classes)) - 1
        if class_index < 0:
            class_index = len(all_classes) - 1

        # 선택된 클래스라면 crop.csv 데이터로 대체
        if class_index in selected_class_indices:
            line = crop_data[i]
        else:
            # 기본 앙상블 데이터 사용
            line = inputs_files_data[0][i]  # 임의로 첫 번째 모델 선택 (필요시 조정)

        output_file.write(line)

print("Ensemble Done!")
print(f"Save Path : {output_file_path}")
