# Create templates
import cv2
import numpy as np
import os

img_path = 'data'
template_path = 'templates'

def extract_template(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    high_HSV = np.array([15, 255, 255]) 
    low_HSV = np.array([0, 50, 50]) 
    cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.inRange(img, low_HSV, high_HSV)

    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped_template = binary_image[y:y+h, x:x+w]
    return cropped_template

for sign in ['bubbles', 'decompress', 'up', 'down', 'hold', 'ok']:
    for i in range(1, 6):
        template_file = f'{img_path}/{sign}-{i}.jpg'
        hand_template = extract_template(template_file)
        h, w = hand_template.shape
        new_dim = (h//3, w//3)
        new_img = cv2.resize(hand_template, new_dim)
        cv2.imwrite(rf'{template_path}/{sign}_template{i}.jpg', new_img)


# resize the template to a smaller dimension
for filename in os.listdir(template_path):        
    img = cv2.imread(rf'{template_path}/{filename}')
    h, w, _ = img.shape
    new_dim = (h//3, w//3)
    new_img = cv2.resize(img, new_dim)
    cv2.imwrite(rf'{template_path}/{filename}', new_img)

# img = cv2.imread(os.path.join(img_path, ok_img), cv2.IMREAD_COLOR)
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()