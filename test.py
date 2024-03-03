import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from main import ImageHandler

class ImageTester:
    def __init__(self, test_folder, image_handler):
        self.test_folder = test_folder
        self.image_handler = image_handler
        self.labels = ['bubbles', 'decompress', 'down', 'hold', 'ok', 'up']
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}
        self.predictions = []
        self.true_labels = []
        self.circularity = []

    def load_and_predict(self):
        for file_name in os.listdir(self.test_folder):
            if file_name.endswith(('.jpg', '.png')):
                true_label = file_name.split('-')[0]  # Assuming format "label-number.jpg/.png"
                true_label_idx = self.label_map[true_label]

                img_path = os.path.join(self.test_folder, file_name)
                img = self.read_image(img_path) 

                _, _, pred = self.image_handler.get_hand(img)

                predicted_label_idx = self.label_map[pred]
                self.predictions.append(predicted_label_idx)
                self.true_labels.append(true_label_idx)

    def read_image(self, img_path):
        img = cv2.imread(img_path)
        return img

    def evaluate(self):
        accuracy = accuracy_score(self.true_labels, self.predictions)

        f1 = f1_score(self.true_labels, self.predictions, average='weighted')

        cm = confusion_matrix(self.true_labels, self.predictions)
        self.plot_confusion_matrix(cm)

        return accuracy, f1, cm

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("result.png")

        # print(self.circularity)
        # plt.show()

image_handler = ImageHandler(5000, 300, 150)
tester = ImageTester(test_folder='data', image_handler=image_handler)
tester.load_and_predict()
accuracy, f1, cm = tester.evaluate()
print(f"Accuracy:\n {accuracy}")
print(f"F1 Score:\n {f1}")
print(f"Confusion Matrix:\n {cm}")
