import numpy as np
import os
import cv2
import pathlib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import RandomRotation, RandomContrast, RandomZoom
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Define the paths to your dataset folders
dataset_path = r'C:\Users\SOHAM\Desktop\Ocular Disease'
normal = pathlib.Path(os.path.join(dataset_path, 'normal'))
glaucoma = pathlib.Path(os.path.join(dataset_path, 'glaucoma'))
DiabRetino = pathlib.Path(os.path.join(dataset_path, 'diabetic_retinopathy'))
cataract = pathlib.Path(os.path.join(dataset_path, 'cataract'))

# Create dictionaries for images and labels
images_dict = {
    "normal": list(normal.glob("*.jpg")),
    "glaucoma": list(glaucoma.glob("*.jpg")),
    "DiabRetino": list(DiabRetino.glob("*.jpeg")),
    "cataract": list(cataract.glob("*.jpg"))
}

labels_dict = {
    "normal": 0,
    "glaucoma": 1,
    "DiabRetino": 2,
    "cataract": 3
}

# Load and preprocess the data
X, y = [], []
for label, images in images_dict.items():
    for image in images:
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (180, 180))
        if image is not None:
            X.append(image)
            y.append(labels_dict[label])

X = np.array(X)
y = np.array(y)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Data augmentation
data_augmentation = Sequential([
    RandomRotation(factor=0.2),
    RandomContrast(factor=0.3),
    RandomZoom(height_factor=0.3, width_factor=0.3),
    RandomZoom(height_factor=0.7, width_factor=0.7)
])

# Build and compile the model
model = Sequential([
    layers.Conv2D(64, (5, 5), padding="same", input_shape=(180, 180, 3), activation="softmax"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(16, (5, 5), padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(8, (5, 5), padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(4, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Train the model
history = model.fit(data_augmentation(X_train), y_train, epochs=150)

model.summary()

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1]*100)

# Generate confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels_dict.keys(), yticklabels=labels_dict.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
class_report = classification_report(y_test, y_pred_classes, target_names=labels_dict.keys())
print(class_report)

# Define the preprocess_input_image and predict_disease functions
def preprocess_input_image(image_path):
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (180, 180))
    input_image = input_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

def predict_disease(input_image_path, model):
    input_image = preprocess_input_image(input_image_path)
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction, axis=1)
    disease_label = {0: 'Normal', 1: 'Glaucoma', 2: 'DiabRetino', 3: 'Cataract'}
    predicted_disease = disease_label[predicted_class[0]]
    return predicted_disease

# Load the trained model
loaded_model = keras.models.load_model('path_to_your_model.h5')

# Example usage:
input_image_path = 'path_to_user_input_image.jpg'  # Replace with the path to the user's input image
predicted_disease = predict_disease(input_image_path, loaded_model)
print(f"The predicted disease for the input image is: {predicted_disease}")
