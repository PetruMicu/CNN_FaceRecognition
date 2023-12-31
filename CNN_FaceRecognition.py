import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to preprocess image for MobileNetV2
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Load MobileNetV2 model without the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to extract features using MobileNetV2
def extract_features(face):
    face_array = preprocess_image(face)
    features = base_model.predict(face_array)
    features = np.reshape(features, (features.shape[0], -1))
    return features

def get_person_name(predicted_label):
    # Define a mapping between labels and person names
    label_to_person = {
        0: "Person1",
        1: "Person2",
        2: "Person3",
        3: "Person4",
        4: "Person5"
    }

    # Return the corresponding person name for the predicted label
    return label_to_person.get(predicted_label, "Unknown")

# Function to detect faces in an image and return the cropped face
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detector model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the first detected face (assuming there is only one face in the image)
        x, y, w, h = faces[0]
        # Crop the face from the image
        face = img[y:y + h, x:x + w]
        # Resize the face to the required input size for MobileNetV2
        face = cv2.resize(face, (224, 224))
        return face
    else:
        return None

# Load your dataset
dataset_path = "./dataset"

# Create embeddings for all images in the reference path only
embeddings = []
labels = []

for label, personality in enumerate(sorted(os.listdir(dataset_path))):
    reference_path = os.path.join(dataset_path, personality, "reference")
    for image_file in os.listdir(reference_path):
        image_path = os.path.join(reference_path, image_file)
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        # Use the detect_face function to get the cropped face
        face = detect_face(img)

        if face is not None:
            # Extract features from the cropped face
            face_embedding = extract_features(face)
            embeddings.append(face_embedding.flatten())  # Flatten the features
            labels.append(label)

# Convert lists to NumPy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Train a simple k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(embeddings, labels)

# Function to perform face recognition
def recognize_face(test_img):
    # Use the detect_face function to get the cropped face
    face = detect_face(test_img)

    if face is not None:
        # Extract features from the cropped face
        test_embedding = extract_features(face).flatten()  # Flatten the features
        predicted_label = knn_classifier.predict(test_embedding.reshape(1, -1))[0]
        return predicted_label
    else:
        return None

# Test the face recognition system for every image in the "test" path
predicted_labels = []
true_labels = []

for label, personality in enumerate(sorted(os.listdir(dataset_path))):
    test_path = os.path.join(dataset_path, personality, "test")
    for image_file in os.listdir(test_path):
        test_image_path = os.path.join(test_path, image_file)
        # Read the image using OpenCV
        test_img = cv2.imread(test_image_path)
        # Use the recognize_face function to get the predicted label
        predicted_label = recognize_face(test_img)

        if predicted_label is not None:
            person_name = get_person_name(predicted_label)
            print("Input Image:", test_image_path)
            print("Predicted Person:", person_name)
            print()

            true_labels.append(label)
            predicted_labels.append(predicted_label)

# Compute accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)
