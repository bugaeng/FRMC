import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

num_age_classes = 7
num_gender_classes = 2

# Define your model class
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Load the pre-trained ResNet18 model
        resnet18 = models.resnet18(pretrained=True)
        # Extract features from the pre-trained model
        self.features = nn.Sequential(*list(resnet18.children())[:-1])  # Remove the last fully connected layer
        # Add your own fully connected layers for age and gender prediction
        self.fc_age = nn.Linear(512, num_age_classes)  # Adjust num_age_classes according to your task
        self.fc_gender = nn.Linear(512, num_gender_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        age_output = self.fc_age(x)
        gender_output = self.fc_gender(x)
        return age_output, gender_output

# Instantiate your model
model = YourModel()

torch.save({'model': model.state_dict()}, 'best_model(52p).pth')

# Load the model
model.load_state_dict(torch.load("best_model(49_6p).pth"), strict=False)
model.eval()

# Initialize OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the image preprocessing function
def preprocess_image(frame):
    # Convert BGR image from OpenCV to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL image
    pil_image = Image.fromarray(rgb_frame)
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# Define the prediction function
def predict_age_and_gender(frame):
    # Image preprocessing
    input_batch = preprocess_image(frame)
    input_var = Variable(input_batch)

    # Pass the image to the model
    with torch.no_grad():
        age_output, gender_output = model(input_var)

    # Convert the outputs to probabilities
    age_probabilities = torch.nn.functional.softmax(age_output[0], dim=0)
    gender_probabilities = torch.nn.functional.softmax(gender_output[0], dim=0)

    # Get the predicted age and gender
    predicted_age_index = torch.argmax(age_probabilities).item()
    predicted_gender_index = torch.argmax(gender_probabilities).item()

    return predicted_age_index, predicted_gender_index

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()

    # Face detection
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a face is detected
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]

        # Predict age and gender from the face region
        predicted_age_index, predicted_gender_index = predict_age_and_gender(face_roi)

        # Display the results on the screen
        cv2.putText(frame, f"Predicted Age Index: {predicted_age_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Predicted Gender Index: {predicted_gender_index}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame on the screen
    cv2.imshow('Age and Gender Prediction', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the camera
cap.release()
cv2.destroyAllWindows()
