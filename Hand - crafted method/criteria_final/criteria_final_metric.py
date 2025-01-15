import torch
import numpy as np
import cv2
import onnxruntime as ort
from scipy.optimize import curve_fit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_segment_path = "segment_model_path"
input_image_path = "image_path"

session = ort.InferenceSession(model_segment_path)


def pixel_count_score(iris_mask): 
    Q_pixel_count = np.sum(iris_mask > 0)
    return float(Q_pixel_count/307200 * 100)


def sharpness_score(image):
    gaussianX = cv2.Sobel(image, cv2.CV_16U, 1, 0)
    gaussianY = cv2.Sobel(image, cv2.CV_16U, 0, 1)
    sharpness_indicator = np.mean(np.sqrt(gaussianX**2 + gaussianY**2))
    return float(sharpness_indicator)

def off_angle_score(light_spots, pupil_center, pupil_radius):
    distances = [np.linalg.norm(np.array(spot) - np.array(pupil_center)) for spot in light_spots]
    max_distance = max(distances, default=0)
    return float(max_distance / pupil_radius) if max_distance > pupil_radius else 0


def dilation_score(iris_param, pupil_param):
    iris_radius = iris_param[2]
    pupil_radius = pupil_param[2]
    all_area = np.pi*(iris_radius**2)
    usable_iris_area = np.pi * (iris_radius**2 - pupil_radius**2)
    if usable_iris_area > 0:
        usable_area_indicator = (usable_iris_area / all_area) * 100
    else:
        usable_area_indicator = 0.0
    return float(usable_area_indicator)


def gray_level_spread_score(img, mask):
    img = img.astype(int)
    img[mask != True] = -1
    usable_pix_num = np.sum(mask)
    if usable_pix_num == 0:
        return 0.0
    gray_level_spread_indicator = 0.0
    for i in range(256):
        p = np.sum(img == i) / usable_pix_num
        if p > 0:
            gray_level_spread_indicator -= p * np.log2(p)
    return float(gray_level_spread_indicator)


def quality_score_fusion(pixel, sharp, angle, dilate, gls):
    a1, b1 = -0.0009808688655225679, 0.014656305101058352
    a2, b2 = -0.00013511397202956458, 0.006720632355328515
    a3, b3, c3 = -1.5009448817622242e-08, 7.213115349032539e-06, -0.0009488171034727981
    a4, b4 = 5.018397140056854e-06, -0.0006690977518425508
    a5, b5 = 0.0013817957959546304, -0.038641146361500026
    c = 0.8143829698067612
    
    return (
        a1 * pixel**2 + b1 * pixel +
        a2 * sharp**2 + b2 * sharp +
        a3 * angle**3 + b3 * angle**2 + c3*angle +
        a4 * dilate**2 + b4 * dilate +
        a5 * gls**2 + b5 * gls +
        c
    )

def process_image_and_calculate_metrics(image):
    original_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    input_size = (640, 480)
    
    def preprocess_image(img):
        img = cv2.resize(img, input_size)
        img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.stack([img, img, img], axis=-1)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img
    
    input_tensor = preprocess_image(original_image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})[0]
    
    def postprocess_output(output):
        segmentation_maps = np.squeeze(output)
        binary_masks = (segmentation_maps > 0.5).astype(np.uint8)
        return binary_masks

    masks = postprocess_output(output)
    mask_iris = masks[1]
    mask_pupil = masks[2]
    mask_iris_resized = cv2.resize(mask_iris, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_pupil_resized = cv2.resize(mask_pupil, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    def define_circle_param(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        return x, y, radius
    
    def find_light_spots(img, threshold=200):
        _, thresholded = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return np.column_stack(np.where(thresholded == 255))
    
    iris_param = define_circle_param(mask_iris_resized)
    pupil_param = define_circle_param(mask_pupil_resized)
    pupil_radius = pupil_param[2]
    pupil_center = (pupil_param[0], pupil_param[1])
    light_spots = find_light_spots(original_image) 

    Q_pixel_value = round(pixel_count_score(mask_iris_resized), 2)
    sharpness_value = round(sharpness_score(original_image), 2)
    off_angle_value = round(off_angle_score(light_spots, pupil_center, pupil_radius), 2)
    dilation_value = round(dilation_score(iris_param, pupil_param), 2)
    gray_level_spread_value = round(gray_level_spread_score(original_image, mask_iris_resized), 2)
    fusion_score = round(quality_score_fusion(Q_pixel_value, sharpness_value, off_angle_value, dilation_value, gray_level_spread_value), 2)

    metrics = {"Pixel Count Score": Q_pixel_value,
               "Sharpness Score": sharpness_value,
               "Off-Angle Score": off_angle_value,
               "Dilation Score": dilation_value,
               "GLS Score": gray_level_spread_value,
               "Fusion Score": fusion_score}
    
    if Q_pixel_value < 4 or sharpness_value < 10 or off_angle_value > 20 or dilation_value < 75 or gray_level_spread_value < 5:
        print("According to iris criteria - based assessment, this image has a poor quality")
    else:
        print("According to iris criteria - based assessment, this image has a good quality")
    return metrics