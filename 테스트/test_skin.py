import cv2
import numpy as np
from keras.models import load_model

# 모델 및 레이블 파일 경로
model_path = './model/keras_model.h5'
labels_path = './model/labels.txt'

# 이미지 파일 경로
image_path = "C:/Users/202-16/Desktop/Face-ID/image/ISIC_0024327.jpg"

# 모델과 레이블 불러오기
model = load_model(model_path)
with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')

# 이미지 불러오기
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 기본적으로 BGR 순서를 사용하므로 RGB로 변환

# 이미지 크기 조정 및 전처리
input_size = (224, 224)  # 모델에 맞는 입력 크기로 조정
resized_image = cv2.resize(image, input_size)
preprocessed_image = resized_image / 255.0  # 모델의 입력에 맞게 정규화

# 모델에 이미지 전달하여 예측 수행
input_data = np.expand_dims(preprocessed_image, axis=0)
predictions = model.predict(input_data)
predicted_class = np.argmax(predictions)
prediction = model.predict(input_data, verbose=0)
probabilities = prediction[0]
confidence_score = prediction[0][np.argmax(prediction)]

# cv2.putText(image, f'Skin disease division: {labels[2:]} Score: {confidence_score:.2f}', (10, 390),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
# 예측한 클래스에 해당하는 라벨과 score 표시
cv2.putText(image, f'Skin disease division: {labels[predicted_class]} Score: {confidence_score:.2f}', (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

 # 각 라벨에 대한 확률도 표시
for i, label in enumerate(labels):
    cv2.putText(image, f'{label[2:]} Probability: {probabilities[i]:.2f}', (10, 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

# 결과 이미지 보여주기
cv2.imshow('Prediction Result', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()