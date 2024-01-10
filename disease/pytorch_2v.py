from flask import Flask, render_template, Response
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

app = Flask(__name__)

num_age_classes = 7
num_gender_classes = 2

# Define your model class
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

# Instantiate your model
model = YourModel()
model.load_state_dict(torch.load("best_model(49_6p).pth"), strict=False)
model.eval()

# Initialize OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(frame):
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

def generate_frames():
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

            cv2.putText(frame, f"Predicted Age Index: {predicted_age_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Predicted Gender Index: {predicted_gender_index}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', cv2.resize(frame, (0,0), fx=0.5, fy=0.5))
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="localhost", port="8000", debug=True)
