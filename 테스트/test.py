import cv2
import dlib
import numpy as np

# app = Flask(__name__)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 이미지 파일 경로
image_path = "/image/face6.jpg"

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지 크기 조절
scale_percent = 1  # 조절하고 싶은 비율로 변경
resized_image = cv2.resize(image, (0, 0), fx=scale_percent, fy=scale_percent)

faces = detector(resized_image)

# 얼굴 특징점 예측
for face in faces:
    landmarks = predictor(resized_image, face)

    # 얼굴 특징 그리기(점)
    # for i in range(68):
    #     x, y = landmarks.part(i).x, landmarks.part(i).y
    #     cv2.circle(resized_image, (x, y), 2, (0, 0, 255), -1)
    for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(resized_image, (x, y), 2, (0, 0, 255), -1)

        # 좌표 출력
            cv2.putText(resized_image, str(i + 1), (x , y ), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # # 코의 중심(34)에서 오른쪽 눈 끝(46), 왼쪽 눈 끝(37)까지의 거리 계산  
        # left_eye_distance = np.sqrt(abs(landmarks.part(34).x - landmarks.part(37).x)**2 + abs(landmarks.part(34).y - landmarks.part(37).y)**2)
        # right_eye_distance = np.sqrt(abs(landmarks.part(34).x - landmarks.part(46).x)**2 + abs(landmarks.part(34).y - landmarks.part(46).y)**2)

        # # 코의 중심(34)에서 오른쪽 입술 끝(55), 왼쪽 입술 끝(49)까지의 거리 계산
        # left_lib_distance = np.sqrt((landmarks.part(34).x - landmarks.part(49).x)**2 + (landmarks.part(34).y - landmarks.part(49).y)**2)
        # right_lib_distance = np.sqrt((landmarks.part(34).x - landmarks.part(55).x)**2 + (landmarks.part(34).y - landmarks.part(55).y)**2)
        
    # 추가
    # 코의 중심
    center_of_nose = (landmarks.part(33).x, landmarks.part(33).y)
    # 왼쪽 눈끝
    left_eye_corner = (landmarks.part(36).x, landmarks.part(36).y)
    # 오른쪽 눈끝
    right_eye_corner = (landmarks.part(45).x, landmarks.part(45).y)
    # 왼쪽 입술
    left_lib_corner = (landmarks.part(48).x, landmarks.part(48).y)
    # 오른쪽 입술
    right_lib_corner = (landmarks.part(54).x, landmarks.part(54).y)

    # 코의 중심(34)에서 오른쪽 입술 끝(55), 왼쪽 입술 끝(49)까지의 거리 계산
    left_lib_distance = np.sqrt((landmarks.part(34).x - landmarks.part(49).x)**2 + (landmarks.part(34).y - landmarks.part(49).y)**2)
    right_lib_distance = np.sqrt((landmarks.part(34).x - landmarks.part(55).x)**2 + (landmarks.part(34).y - landmarks.part(55).y)**2)

    # 코의 중심(34)에서 오른쪽 눈 끝(46), 왼쪽 눈 끝(37)까지의 거리 계산  
    left_eye_distance = np.sqrt((left_eye_corner[0] - center_of_nose[0])**2 + (left_eye_corner[1] - center_of_nose[1])**2)
    right_eye_distance = np.sqrt((right_eye_corner[0] - center_of_nose[0])**2 + (right_eye_corner[1] - center_of_nose[1])**2)

    # 코에 중심부터 눈끝까지의 직선 그리기
    cv2.line(resized_image, center_of_nose, left_eye_corner, (255, 0, 0), 2)
    cv2.line(resized_image, center_of_nose, right_eye_corner, (255, 0, 0), 2)

    # 코에 중심부터 입술까지의 직선 그리기
    cv2.line(resized_image, center_of_nose, left_lib_corner, (255, 0, 0), 2)
    cv2.line(resized_image, center_of_nose, right_lib_corner, (255, 0, 0), 2)

    # 거리 출력
    cv2.putText(resized_image, f'Left lip Distance(49): {left_lib_distance:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(resized_image, f'Right lip Distance(55): {right_lib_distance:.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(resized_image, f'Left eye Distance(37): {left_eye_distance:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(resized_image, f'Right eye Distance(46): {right_eye_distance:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 대칭 여부 판단
    # symmetric_lip = "Symmetric" if np.isclose(left_lib_distance, right_lib_distance, rtol=1e-05, atol=1e-08) else "Asymmetric"
    # symmetric_eye = "Symmetric" if np.isclose(left_eye_distance, right_eye_distance, rtol=1e-05, atol=1e-08) else "Asymmetric"
    # cv2.putText(resized_image, f'Lip Symmetry: {symmetric_lip}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.putText(resized_image, f'Eye Symmetry: {symmetric_eye}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # 정의(압축해서 표현하기)
    Lip_Asymmetric_Value = abs(left_lib_distance - right_lib_distance)
    Eye_Asymmetric_Value = abs(left_eye_distance - right_eye_distance)
    total_Value = abs(left_lib_distance - right_lib_distance) + abs(left_eye_distance - right_eye_distance)
    
    # 토탈 오차 값
    cv2.putText(resized_image, f'Lip Asymmetric Value : {Lip_Asymmetric_Value:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(resized_image, f'Eye Asymmetric Value : {Eye_Asymmetric_Value:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(resized_image, f'total Value : {total_Value:.2f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    if ((total_Value) > 20 ): # or Lip_Asymmetric_Value >= 5 일단 제외
        cv2.putText(resized_image, "Result : Asymmetric", (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else :
        cv2.putText(resized_image, "Result : Symmetric", (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# 이미지 저장 (옵션)
# cv2.imwrite("path/to/your/output_image.jpg", image)

cv2.imshow("Facial Landmarks", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
