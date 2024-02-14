import cv2
import numpy as np
import random
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
import os

img_path = 'data'
template_path = 'templates'
template_path = 'old_templates'
# for filename in os.listdir(template_path):        
#     print(filename)


class ImageHandler:
    def __init__(self, min_area, min_length, distance, draw_type=0, max_area=100000, max_length=100000):
        """
        :param min_area: Minimum area for contour detection
        :param min_length: Minimum perimeter for contour detection
        :param distance: Distance threshold for point detection
        :param draw_type: Type of drawing to apply on detected contours (0 for polylines, 1 for lines)
        :param max_area: Maximum area for contour detection
        :param max_length: Maximum perimeter for contour detection
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_length = min_length
        self.max_length = max_length
        self.distance = distance
        self.points_list = []   # points of detected contours
        self.high_HSV = np.array([15, 255, 255])    # Upper HSV threshold for filtering
        self.low_HSV = np.array([0, 50, 50])        # Lower HSV threshold for filtering
        self.draw_type = draw_type
        self.low_skin = np.array([20, 40, 75], dtype="uint8")
        self.high_skin = np.array([255, 255, 180], dtype="uint8")
        # self.low_skin = np.array([0, 48, 80], dtype="uint8")
        # self.high_skin = np.array([20, 255, 255], dtype="uint8")
        self.src_img = None
        self.img = None
        self.output_img = None

    # Update the distance threshold for point detection
    def change_distance(self, distance):
        self.distance = distance

    # Update the HSV color filtering bounds
    def change_HSV(self, low_HSV, high_HSV):
        self.low_HSV = low_HSV
        self.high_HSV = high_HSV

    # Resize the image
    def resize_img(self, img, size):
        size = [size[1], size[0], size[2]]  # [h, w, c]
        mask = np.zeros(size, dtype=np.uint8)
        h, w = img.shape[0:2]
    
        dwh = min([size[0] / h, size[1] / w])
        img = cv2.resize(img, None, fx=dwh, fy=dwh)

        if h > w:
            dxy = int((size[1] - img.shape[1]) / 2)
            mask[:, dxy:img.shape[1] + dxy, :] = img
        else:
            dxy = int((size[0] - img.shape[0]) / 2)
            mask[dxy:img.shape[0] + dxy, :, :] = img
        
        # print(mask.shape)
        return mask

    # Create the binary image
    def preprocess_with_HSV(self):
        self.img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2HSV)
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
        self.img = cv2.inRange(self.img, self.low_HSV, self.high_HSV)
        kernel = np.ones((3, 3), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_DILATE, kernel)

    def preprocess_with_skin_color(self):
        self.img = cv2.GaussianBlur(self.src_img, (3, 3), 0)
        self.img = cv2.inRange(self.img, self.low_skin, self.high_skin)
        kernel = np.ones((100, 100), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_DILATE, kernel)
    
    # Process the image for contour detection
    def process(self, img=None):
        if img is not None:
            self.img = img

        contours, _ = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, True)
            if self.max_area > area > self.min_area and self.max_length > length > self.min_length:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                self.points_list = cv2.approxPolyDP(contour, epsilon, True)
                self.points_list = self.points_list.reshape(len(self.points_list), 2)
                self.points_list = np.array(self.points_list, dtype=np.int32)

                if self.draw_type == 0:
                    b = random.randint(0, 255)
                    g = random.randint(0, 255)
                    r = random.randint(0, 255)
                    cv2.polylines(self.output_img, [self.points_list], True, [b, g, r], 4, 16)
        return self.img

    def get_distance(self, pt1, pt2):
        return ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
    
    def calculate_circularity(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        return (4 * np.pi * area) / (perimeter ** 2)
    
    def detect_by_circularity(self):
        gesture = ''
        circularity = 0
        contours, _ = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, True)
            if self.max_area > area > self.min_area and self.max_length > length > self.min_length:
                circularity = self.calculate_circularity(contour)
                gesture = round(circularity, 2)
                # print(circularity)
        
        if circularity >= 0.41:
            gesture = ['hold']
        elif circularity < 0.41 and circularity >= 0.31:
            gesture = ['bubbles']
        elif circularity < 0.31 and circularity >= 0.22:
            gesture = ['down'] 
        elif circularity < 0.22:
            gesture = ['decompress', 'ok', 'up']
        # cv2.putText(self.output_img, str(gesture), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        return gesture

    def detect_by_template(self):
        predicted_handshape = [0]*6
        ind = 0
        for filename in os.listdir(template_path):        
            file_path = os.path.join('templates', filename)
            template = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # print(self.img.shape)
            # print(template.shape)
            result = cv2.matchTemplate(self.img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            predicted_handshape[ind//5] += max_val
            ind += 1

        answer = ['bubbles', 'decompress', 'down', 'hold', 'ok', 'up']
        max_prob = max(predicted_handshape)
        ind = int(predicted_handshape.index(max_prob))
        gesture = answer[ind]
        return gesture
    
    def detect_by_finger(self):
        num = 0
        if np.any(self.points_list):
            max_index = np.argmax(self.points_list, axis=0)
            for point in self.points_list:
                distance = self.get_distance(self.points_list[max_index[1], :], point)
                if distance > self.distance:
                    if self.draw_type == 1:
                        b = random.randint(0, 255)
                        g = random.randint(0, 255)
                        r = random.randint(0, 255)
                        cv2.line(self.output_img,self.points_list[max_index[1], :], point, [b, g, r], 4, 16)
                    num += 1
        return num

    def detect(self):
        circularity = self.detect_by_circularity()
        template = self.detect_by_template()
        finger = self.detect_by_finger()

        return (circularity, template, finger)
 
    # Entrance
    def get_hand(self, img):
        self.img = img  # [h, w, c]
        if self.img.shape[0] != 480 and self.img.shape[1] != 480:
            self.img = self.resize_img(img, [480, 480, 3])
        self.src_img = np.copy(self.img)
        self.output_img = np.copy(self.img)

        # self.preprocess_with_skin_color()
        self.preprocess_with_HSV()

        self.img = self.process()
        (circularity, template, finger) = self.detect()

        gesture = str((circularity, template, finger))
        cv2.putText(self.output_img, gesture, [10, 50], cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], thickness=2)

        gesture = (circularity, template, finger)
        return self.output_img, self.img, gesture

    

class GUI:
    def __init__(self):
        self.image_handler = ImageHandler(20000, 1000, 100)
        self.result_text = ''
        self.video = ''
        self.after = ''
        self.file_type = ['.mp4', '.png', '.jpg']
        self.file_name = ''

        self.root = tk.Tk()
        self.root.geometry('1000x700')
        self.root.title('Scuba Hand Singal Recognition')
        self.root.resizable(width=False, height=False)

        self.img1_label = tk.Label(self.root, text='', bg='white', bd=10)
        self.img1_label.place(x=340, y=20, width=640, height=480)

        self.img2_label = tk.Label(self.root, text='', bg='white', bd=10)
        self.img2_label.place(x=20, y=220, width=250, height=400)

        self.select_file_button = tk.Button(self.root, text='Select File', command=self.select_file, font=('Arial', 20), bg='green', bd=10)
        self.select_file_button.place(x=20, y=20, width=250, height=50)

        self.open_file_button = tk.Button(self.root, text='Open', command=self.open_file, font=('Arial', 20), bg='blue', bd=10)
        self.open_file_button.place(x=20, y=90, width=250, height=50)

        self.open_camera_button = tk.Button(self.root, text='Open Camera', command=self.toggle_camera, font=('Arial', 20), bg='white', bd=10)
        self.open_camera_button.place(x=20, y=160, width=250, height=50)

        self.result_var = tk.StringVar(self.root, value='')
        self.result_entry = tk.Entry(self.root, textvariable=self.result_var, state='readonly', font=('Arial', 20), bg='white', bd=10)
        self.result_entry.place(x=340, y=520, width=640, height=140)

        # self.distance_threshold_var = tk.IntVar(self.root)
        # self.distance_scale = tk.Scale(self.root, label='Distance Threshold', from_=0, to=800, 
        #                                resolution=1, orient=tk.HORIZONTAL, tickinterval=200, variable=self.distance_threshold_var, bg='white', bd=10)
        # self.distance_scale.place(x=20, y=620, width=250)
        self.update_result()

    def toggle_camera(self):
        if self.video:
            self.video.release()
        self.video = cv2.VideoCapture(0)
        if self.after:
            self.root.after_cancel(self.after)
        self.open_video()

    def open_video(self):
        res, img=self.video.read()
        if res == True and np.any(img):
            img1, img2, self.result_text = self.image_handler.get_hand(img)
            self.display_image1(img1)
            self.display_image2(img2)
        self.after = self.root.after(10, self.open_video)

    def open_file(self):
        if not self.file_name:
            messagebox.showerror(title='Warning', message='Please select a video or image file.')
        elif self.file_type[0] in self.file_name:
            if self.video:
                self.video.release()
            self.video = cv2.VideoCapture(self.file_name)
            if self.after:
                self.root.after_cancel(self.after)
            self.open_video()
        else:
            if self.video:
                self.video.release()
            img = cv2.imread(self.file_name)
            img1, img2, self.result_text = self.image_handler.get_hand(img)
            self.display_image1(img1)
            self.display_image2(img2)

    def select_file(self):
        self.file_name = filedialog.askopenfilename()
        num = 0
        for path in self.file_type:
            if path not in self.file_name:
                    num += 1
        if num == 3:
            messagebox.showerror(title='Warning', message='Please select a video or image file.')      

    def update_result(self):
        # distance = self.distance_threshold_var.get()
        # self.image_handler.change_distance(distance)
        self.result_var.set(f'{self.result_text}')
        self.root.after(10, self.update_result)

    def display_image1(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.img1_label.image = img_tk
        self.img1_label['image'] = img_tk

    def display_image2(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = self.image_handler.resize_img(img, [250, 400, 3])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.img2_label.image = img_tk
        self.img2_label['image'] = img_tk

    def run(self):
        # messagebox.showinfo('Notice', message='Adjust the distance threshold for better performance.')
        self.root.mainloop()

    def close(self):
        if self.video:
            self.video.release()


if __name__ == "__main__":
    app = GUI()
    app.run()
    app.close()
