import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to preprocess image for MobileNetV2
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Load MobileNetV2 model without the top classification layer

# Function to extract features using MobileNetV2
def extract_features(img_path):
    img_array = preprocess_image(img_path)
    features = base_model.predict(img_array)
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

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load your dataset
dataset_path = "./dataset"

# Create embeddings for all images in the reference path only
embeddings = []
labels = []

for label, personality in enumerate(sorted(os.listdir(dataset_path))):
    reference_path = os.path.join(dataset_path, personality, "reference")
    for image_file in os.listdir(reference_path):
        image_path = os.path.join(reference_path, image_file)
        face_embedding = extract_features(image_path)
        embeddings.append(face_embedding.flatten())  # Flatten the features
        labels.append(label)

# Convert lists to NumPy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Train a simple k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(embeddings, labels)

# Function to perform face recognition
def recognize_face(test_image_path):
    test_embedding = extract_features(test_image_path).flatten()  # Flatten the features
    predicted_label = knn_classifier.predict(test_embedding.reshape(1, -1))[0]
    return predicted_label

# Test the face recognition system for every image in the "test" path
predicted_labels = []
true_labels = []

for label, personality in enumerate(sorted(os.listdir(dataset_path))):
    test_path = os.path.join(dataset_path, personality, "test")
    for image_file in os.listdir(test_path):
        test_image_path = os.path.join(test_path, image_file)
        predicted_label = recognize_face(test_image_path)

        person_name = get_person_name(predicted_label)
        print("Input Image:", test_image_path)
        print("Predicted Person:", person_name)
        print()

        true_labels.append(label)
        predicted_labels.append(predicted_label)

# Compute accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)
