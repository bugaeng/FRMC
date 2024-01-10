from flask import Flask, render_template, request, redirect, url_for, Response
import os
import imutils

from time import sleep
import cv2
import dlib
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from flask_cors import CORS
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
cors = CORS(app, resources={r"/Facial_palsy/*": {"origins": "*"}})
cors = CORS(app, resources={r"/Skin_disease/*": {"origins": "*"}})
cors = CORS(app, resources={r"/Capture/*": {"origins": "*"}})
cors = CORS(app, resources={r"/prediction/*": {"origins": "*"}})

# dlib의 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()

# 얼굴 랜드마크 예측 모델 로드
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   

# capture = cv2.VideoCapture('http://192.168.0.5:81/stream')
capture = cv2.VideoCapture(0)
sleep(2.0)

Lip_Asymmetric_Value = 0
Eye_Asymmetric_Value = 0
total_Value = 0

predicted_label_n = 0
predicted_label_h = 0
confidence_score_v = 0

left_lib_distance_d = 0
right_lib_distance_d = 0
left_eye_distance_d = 0
right_eye_distance_d = 0

label0 = 0
label1 = 0
label2 = 0
label3 = 0
label4 = 0
label5 = 0
label6 = 0

probabilities0 = 0
probabilities1 = 0
probabilities2 = 0
probabilities3 = 0
probabilities4 = 0
probabilities5 = 0
probabilities6 = 0


# kang
gender_text = ""
age_text = ""

# 얼굴 비대칭
def generate():
    global Lip_Asymmetric_Value
    global Eye_Asymmetric_Value
    global total_Value
    global left_lib_distance_d 
    global right_lib_distance_d
    global left_eye_distance_d
    global right_eye_distance_d
    while True:
        ret, frame = capture.read()  
        frame = cv2.resize(frame, (600, 400))

        rgb_small_frame = frame[:, :, ::-1]
        # dlib을 사용하여 얼굴 감지
        faces = detector(rgb_small_frame)
        # 얼굴 특징 그리기(선)
        for face in faces:
            # 얼굴의 랜드마크를 예측
            landmarks = predictor(rgb_small_frame, face)

            # # 얼굴 특징을 선으로 그리기
            # for i in range(1, 68):
            #     cv2.line(frame, (landmarks.part(i - 1).x , landmarks.part(i - 1).y ),
            #              (landmarks.part(i).x , landmarks.part(i).y ), (255, 0, 0), 1)
            # 얼굴 특징 그리기(점)
            for i in range(68):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 0, 255), -1)

            # 좌표 출력
                # cv2.putText(frame, str(i + 1), (x , y ), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

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
            cv2.line(frame, center_of_nose, left_eye_corner, (255, 0, 0), 2)
            cv2.line(frame, center_of_nose, right_eye_corner, (255, 0, 0), 2)

            # 코에 중심부터 입술까지의 직선 그리기
            cv2.line(frame, center_of_nose, left_lib_corner, (255, 0, 0), 2)
            cv2.line(frame, center_of_nose, right_lib_corner, (255, 0, 0), 2)

            # 거리 출력
            cv2.putText(frame, f'Left lip Distance(49): {left_lib_distance:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Right lip Distance(55): {right_lib_distance:.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Left eye Distance(37): {left_eye_distance:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Right eye Distance(46): {right_eye_distance:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 대칭 여부 판단
            # symmetric_lip = "Symmetric" if np.isclose(left_lib_distance, right_lib_distance, rtol=1e-05, atol=1e-08) else "Asymmetric"
            # symmetric_eye = "Symmetric" if np.isclose(left_eye_distance, right_eye_distance, rtol=1e-05, atol=1e-08) else "Asymmetric"
            # cv2.putText(frame, f'Lip Symmetry: {symmetric_lip}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(frame, f'Eye Symmetry: {symmetric_eye}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # 정의(압축해서 표현하기)
            Lip_Asymmetric_Value = round(abs(left_lib_distance - right_lib_distance),4)
            Eye_Asymmetric_Value = round(abs(left_eye_distance - right_eye_distance),4)
            total_Value = round(abs(left_lib_distance - right_lib_distance) + abs(left_eye_distance - right_eye_distance),4)
            left_lib_distance_d = round(np.sqrt((landmarks.part(34).x - landmarks.part(49).x)**2 + (landmarks.part(34).y - landmarks.part(49).y)**2),4)
            right_lib_distance_d = round(np.sqrt((landmarks.part(34).x - landmarks.part(55).x)**2 + (landmarks.part(34).y - landmarks.part(55).y)**2),4)
            left_eye_distance_d = round(np.sqrt((left_eye_corner[0] - center_of_nose[0])**2 + (left_eye_corner[1] - center_of_nose[1])**2),4)
            right_eye_distance_d = round(np.sqrt((right_eye_corner[0] - center_of_nose[0])**2 + (right_eye_corner[1] - center_of_nose[1])**2),4)
            
            # 토탈 오차 값
            cv2.putText(frame, f'Lip Asymmetric Value : {Lip_Asymmetric_Value:.2f}', (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Eye Asymmetric Value : {Eye_Asymmetric_Value:.2f}', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'total Value : {total_Value:.2f}', (260, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            if ((total_Value) > 10 ): # or Lip_Asymmetric_Value >= 5 일단 제외
                cv2.putText(frame, "Result : Asymmetric", (420,390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else :
                cv2.putText(frame, "Result : Symmetric", (420,390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if ret :
            _, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes()  
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 피부 병변 모델 로드
model = load_model("keras_model.h5")
with open("labels.txt", "r") as file:
    labels = file.read().splitlines()

def generate2():
    global predicted_label_n
    global predicted_label_h
    global confidence_score_v
    global label0
    global label1
    global label2
    global label3
    global label4
    global label5
    global label6
    global probabilities0
    global probabilities1
    global probabilities2
    global probabilities3
    global probabilities4
    global probabilities5
    global probabilities6
    while True:
        ret, frame = capture.read()
        frame = cv2.resize(frame, (600, 400))

        # 피부 병변 인식 모델에 프레임을 위한 전처리
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # 로드한 모델을 사용하여 예측
        prediction = model.predict(img_array, verbose=0)

        # 예측된 레이블 얻기
        predicted_label = labels[np.argmax(prediction)]

        # 각 라벨에 대한 확률 얻기
        probabilities = prediction[0]

        # # 프레임에 예측된 레이블 표시
        # cv2.putText(frame, f'Skin disease division: {predicted_label}', (10, 240),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        
        # 예측된 레이블에 해당하는 확률 얻기
        confidence_score = prediction[0][np.argmax(prediction)]

        # 프레임에 예측된 레이블과 확률 표시
        cv2.putText(frame, f'Skin disease division: {predicted_label[2:]} Score: {confidence_score:.2f}', (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame, f'', (10, 260),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        predicted_label_n = predicted_label[:1]
        predicted_label_h = predicted_label[2:]
        confidence_score_v = round(confidence_score, 2)


        # 각 라벨에 대한 확률도 표시
        for i, label in enumerate(labels):
            cv2.putText(frame, f'{label[2:]} Probability: {probabilities[i]:.2f}', (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        label0 = labels[0]
        label1 = labels[1]
        label2 = labels[2]
        label3 = labels[3]
        label4 = labels[4]
        label5 = labels[5]
        label6 = labels[6]

        probabilities0 = probabilities[0]
        probabilities1 = probabilities[1]
        probabilities2 = probabilities[2]
        probabilities3 = probabilities[3]
        probabilities4 = probabilities[4]
        probabilities5 = probabilities[5]
        probabilities6 = probabilities[6]

        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 일반 캠
def generate3():
    while True:
        ret, frame = capture.read()
        frame = cv2.resize(frame, (600,400))

        if ret :
            _, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes()  
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


num_age_classes = 7
num_gender_classes = 2

def generate4():

    global gender_text
    global age_text

    class YourModel(nn.Module):
        def __init__(self):
            super(YourModel, self).__init__()
            resnet18 = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(resnet18.children())[:-1])
            self.fc_age = nn.Linear(512, num_age_classes)
            self.fc_gender = nn.Linear(512, num_gender_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            age_output = self.fc_age(x)
            gender_output = self.fc_gender(x)
            return age_output, gender_output

    model = YourModel()
    model.load_state_dict(torch.load("best_model(49_6p).pth"), strict=False)
    model.eval()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def preprocess_image(frame):
        frame = cv2.resize(frame, (600, 400))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def predict_age_and_gender(frame):
        input_batch = preprocess_image(frame)
        input_var = Variable(input_batch)

        with torch.no_grad():
            age_output, gender_output = model(input_var)

        age_probabilities = torch.nn.functional.softmax(age_output[0], dim=0)
        gender_probabilities = torch.nn.functional.softmax(gender_output[0], dim=0)

        predicted_age_index = torch.argmax(age_probabilities).item()
        predicted_gender_index = torch.argmax(gender_probabilities).item()

        return predicted_age_index, predicted_gender_index

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]

            predicted_age_index, predicted_gender_index = predict_age_and_gender(face_roi)

            gender_label = "Male" if predicted_gender_index == 0 else "Female"
            age_value = (predicted_age_index + 1) * 10


            gender_text = gender_label
            age_text = age_value

            if gender_label == "Male":
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            text_age = f"Age: {age_value}s"
            text_width_age, text_height_age = cv2.getTextSize(text_age, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            background_width_age = text_width_age + 10
            cv2.rectangle(frame, (x, y + h + 10), (x + background_width_age, y + h + text_height_age + 20), (255, 0, 0), -1)

            text_gender = f"Gender: {gender_label}"
            text_width_gender, text_height_gender = cv2.getTextSize(text_gender, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            background_width_gender = text_width_gender + 10
            cv2.rectangle(frame, (x, y + h + text_height_age + 30), (x + background_width_gender, y + h + text_height_age + text_height_gender + 40), (255, 0, 0), -1)

            cv2.putText(frame, f"Age: {age_value}s", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Gender: {gender_label}", (x, y + h + text_height_age + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', cv2.resize(frame, (0,0), fx=0.5, fy=0.5))
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 얼굴 비대칭 검사(dlib 라이브러리 사용)
@app.route('/Facial_palsy')
def Facial_palsy():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 얼굴 비대칭 검사(벨류값)
@app.route('/Facial_palsy_Lip_Value')
def Facial_palsy_Lip_Value():
    global Lip_Asymmetric_Value
    return Response(str(Lip_Asymmetric_Value), mimetype='text/plain')

@app.route('/Facial_palsy_Eye_Value')
def Facial_palsy_Eye_Value():
    global Eye_Asymmetric_Value
    return Response(str(Eye_Asymmetric_Value), mimetype='text/plain')

@app.route('/Facial_palsy_Total_Value')
def Facial_palsy_Total_Value():
    global total_Value
    return Response(str(total_Value), mimetype='text/plain')

@app.route('/Facial_palsy_left_lib_distance_d')
def left_lib_distance_d():
    global left_lib_distance_d
    return Response(str(left_lib_distance_d), mimetype='text/plain')

@app.route('/Facial_palsy_right_lib_distance_d')
def right_lib_distance_d():
    global right_lib_distance_d
    return Response(str(right_lib_distance_d), mimetype='text/plain')

@app.route('/Facial_palsy_left_eye_distance_d')
def left_eye_distance_d():
    global left_eye_distance_d
    return Response(str(left_eye_distance_d), mimetype='text/plain')

@app.route('/Facial_palsy_right_eye_distance_d')
def right_eye_distance_d():
    global right_eye_distance_d
    return Response(str(right_eye_distance_d), mimetype='text/plain')



# 피부 병변 검사(모델 사용)
@app.route('/Skin_disease')
def Skin_disease():
    return Response(generate2(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 피부 병변 검사(벨류값)
@app.route('/Skin_disease_label_n')
def Skin_disease_label_n():
    global predicted_label_n
    return Response(str(predicted_label_n), mimetype='text/plain')

@app.route('/Skin_disease_label_h')
def Skin_disease_label_h():
    global predicted_label_h
    return Response(str(predicted_label_h), mimetype='text/plain')

@app.route('/Skin_disease_score_v')
def Skin_disease_score_v():
    global confidence_score_v
    return Response(str(confidence_score_v), mimetype='text/plain')

# 피부 병변 라벨셋
@app.route('/Skin_disease_label0')
def label0():
    global label0
    return Response(str(label0[2:]), mimetype='text/plain')
@app.route('/Skin_disease_label1')
def label1():
    global label1
    return Response(str(label1[2:]), mimetype='text/plain')
@app.route('/Skin_disease_label2')
def label2():
    global label2
    return Response(str(label2[2:]), mimetype='text/plain')
@app.route('/Skin_disease_label3')
def label3():
    global label3
    return Response(str(label3[2:]), mimetype='text/plain')
@app.route('/Skin_disease_label4')
def label4():
    global label4
    return Response(str(label4[2:]), mimetype='text/plain')
@app.route('/Skin_disease_label5')
def label5():
    global label5
    return Response(str(label5[2:]), mimetype='text/plain')
@app.route('/Skin_disease_label6')
def label6():
    global label6
    return Response(str(label6[2:]), mimetype='text/plain')

# 피부 병변 라벨벨류셋
@app.route('/Skin_disease_probabilities0')
def probabilities0():
    global probabilities0
    return Response(str(round(probabilities0,4)), mimetype='text/plain')
@app.route('/Skin_disease_probabilities1')
def probabilities1():
    global probabilities1
    return Response(str(round(probabilities1,4)), mimetype='text/plain')
@app.route('/Skin_disease_probabilities2')
def probabilities2():
    global probabilities2
    return Response(str(round(probabilities2,4)), mimetype='text/plain')
@app.route('/Skin_disease_probabilities3')
def probabilities3():
    global probabilities3
    return Response(str(round(probabilities3,4)), mimetype='text/plain')
@app.route('/Skin_disease_probabilities4')
def probabilities4():
    global probabilities4
    return Response(str(round(probabilities4,4)), mimetype='text/plain')
@app.route('/Skin_disease_probabilities5')
def probabilities5():
    global probabilities5
    return Response(str(round(probabilities5,4)), mimetype='text/plain')
@app.route('/Skin_disease_probabilities6')
def probabilities6():
    global probabilities6
    return Response(str(round(probabilities6,4)), mimetype='text/plain')

# 캡쳐
@app.route('/Capture')
def Capture():
    return Response(generate3(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 나이 예측
@app.route('/prediction')
def prediction():
    return Response(generate4(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction_text')
def prediction_text():
    global age_text
    global gender_text
    return Response(f"{age_text} {gender_text}", mimetype='text/plain')


if __name__ == "__main__":
    app.run(host="localhost", port="8005", debug=True)