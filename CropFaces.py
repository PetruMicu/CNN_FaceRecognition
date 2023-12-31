import os
import cv2

def print_cropped_face(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Function to detect faces in an image and return the cropped face
    def detect_face(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the pre-trained face detector model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        cropped_faces = []
        for (x, y, w, h) in faces:
            # Crop the face from the image
            face = img[y:y + h, x:x + w]
            # Resize the face to the required input size for MobileNetV2
            face = cv2.resize(face, (224, 224))
            cropped_faces.append(face)

        return cropped_faces

    # Use the detect_face function to get the cropped faces
    cropped_faces = detect_face(img)

    if cropped_faces:
        face = cropped_faces[0]
        print(image_path)
        cv2.imshow(f"Cropped Face", face)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No faces detected in the image.")

# Example usage:
dataset_path = "./images"
for label, personality in enumerate(sorted(os.listdir(dataset_path))):
    path = os.path.join(dataset_path, "person4")
    for image_file in os.listdir(path):
        image_path = os.path.join(path, image_file)
        print_cropped_face(image_path)