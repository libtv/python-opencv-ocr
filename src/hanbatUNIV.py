import cv2
import numpy as np
import yaml
import datetime
import pytesseract as ocr
from PIL import Image

configs = None
ocr.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def read_configs(config_file):
    # read contents from .yam config file
    with open(config_file, 'r') as yml_file:
        configurations = yaml.load(yml_file, Loader=yaml.FullLoader)  # use 'yaml' package to read .yml file

    global configs  # global var : configs
    configs = configurations  # set configs
    return configurations  # return read configurations

def resize(image, flag=-1):
    # get configs
    global configs
    standard_height = configs['resize_origin']['standard_height']
    standard_width = configs['resize_origin']['standard_width']
    # get image size
    height, width = image.shape[:2]
    image_copy = image.copy()
    # print original size (width, height)
    print("origin (width : " + str(width) + ", height : " + str(height) + ")")
    rate = 1  # default
    if (flag > 0 and height < standard_height) or (flag < 0 and height > standard_height):  # Resize based on height
        rate = standard_height / height
    elif (flag > 0 and width < standard_width) or (flag < 0 and height > standard_height):  # Resize based on width
        rate = standard_width / width
    # resize
    w = round(width * rate)  # should be integer
    h = round(height * rate)  # should be integer
    image_copy = cv2.resize(image_copy, (w, h))
    # print modified size (width, height)
    print("after resize : (width : " + str(w) + ", height : " + str(h) + ")")
    return image_copy


def open_original(file_path):
    image_origin = cv2.imread(file_path)  # read image from file
    return image_origin


def get_gray(image_origin):
    copy = image_origin.copy()  # copy the image to be processed
    image_grey = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)  # apply gray-scale to the image
    return image_grey


def get_canny(image_gray):
    copy = image_gray.copy()
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(copy, (kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    return edges


def get_gradient(image_gray):
    copy = image_gray.copy()  # copy the image to be processed
    # get configs
    global configs
    kernel_size_row = configs['gradient']['kernel_size_row']
    kernel_size_col = configs['gradient']['kernel_size_col']
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_row, kernel_size_col))
    # morph gradient
    image_gradient = cv2.morphologyEx(copy, cv2.MORPH_GRADIENT, kernel)
    return image_gradient


def remove_long_line(image_binary):
    copy = image_binary.copy()  # copy the image to be processed
    # get configs
    global configs
    threshold = configs['remove_line']['threshold']
    min_line_length = configs['remove_line']['min_line_length']
    max_line_gap = configs['remove_line']['max_line_gap']

    # find and remove lines
    lines = cv2.HoughLinesP(copy, 1, np.pi / 180, threshold, np.array([]), min_line_length, max_line_gap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # get end point of line : ( (x1, y1) , (x2, y2) )
            # slop = 0
            # if x2 != x1:
            #     slop = abs((y2-y1) / (x2-x1))
            # if slop < 0.5 or slop > 50 or x2 == x1:  # only vertical or parallel lines.
            # remove line drawing black line
            cv2.line(copy, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return copy


def get_threshold(image_gray):
    copy = image_gray.copy()  # copy the image to be processed
    # get configs
    global configs
    mode = configs['threshold']['mode']  # get threshold mode (mean or gaussian or global)
    block_size = configs['threshold']['block_size']
    subtract_val = configs['threshold']['subtract_val']

    if mode == 'mean':  # adaptive threshold - mean
        image_threshold = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV, block_size, subtract_val)
    elif mode == 'gaussian':  # adaptive threshold - gaussian
        image_threshold = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, block_size, subtract_val)
    else:  # (mode == 'global') global threshold - otsu's binary operation
        image_threshold = get_otsu_threshold(copy)

    return image_threshold  # Returns the image with the threshold applied.


def get_global_threshold(image_gray, threshold_value=130):
    copy = image_gray.copy()  # copy the image to be processed
    _, binary_image = cv2.threshold(copy, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image


def get_otsu_threshold(image_gray):
    copy = image_gray.copy()  # copy the image to be processed
    blur = cv2.GaussianBlur(copy, (5, 5), 0)  # Gaussian blur 를 통해 noise 를 제거한 후
    # global threshold with otsu's binarization
    ret3, image_otsu = cv2.threshold(copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image_otsu


def get_closing(image_gray):
    copy = image_gray.copy()  # copy the image to be processed
    # get configs
    global configs
    kernel_size_row = configs['close']['kernel_size_row']
    kernel_size_col = configs['close']['kernel_size_col']
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_row, kernel_size_col))
    # closing (dilation and erosion)
    image_close = cv2.morphologyEx(copy, cv2.MORPH_CLOSE, kernel)
    return image_close


def get_contours(image):
    global configs
    retrieve_mode = configs['contour']['retrieve_mode']  # integer value
    approx_method = configs['contour']['approx_method']  # integer value
    # find contours from the image
    contours, _ = cv2.findContours(image, retrieve_mode, approx_method)
    return contours


def draw_contour_rect(image_origin, contours):
    rgb_copy = image_origin.copy()  # copy the image to be processed
    # get configs
    global configs
    min_width = configs['contour']['min_width']
    min_height = configs['contour']['min_height']
    # Draw bounding rectangles
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)  # top-left vertex coordinates (x,y) , width, height
        # Draw screenshot that are larger than the standard size
        if width > min_width and height > min_height:
            cv2.rectangle(rgb_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return rgb_copy


def get_cropped_images(image_origin, contours):
    image_copy = image_origin.copy()  # copy the image to be processed
    # get configs
    global configs
    min_width = configs['contour']['min_width']
    min_height = configs['contour']['min_height']
    padding = 8  # to give the padding when cropping the screenshot
    origin_height, origin_width = image_copy.shape[:2]  # get image size
    cropped_images = []  # list to save the crop image.

    for contour in contours:  # Crop the screenshot with on bounding rectangles of contours
        x, y, width, height = cv2.boundingRect(contour)  # top-left vertex coordinates (x,y) , width, height
        # screenshot that are larger than the standard size
        if width > min_width and height > min_height:
            # The range of row to crop (with padding)
            row_from = (y - padding) if (y - padding) > 0 else y
            row_to = (y + height + padding) if (y + height + padding) < origin_height else y + height
            # The range of column to crop (with padding)
            col_from = (x - padding) if (x - padding) > 0 else x
            col_to = (x + width + padding) if (x + width + padding) < origin_width else x + width
            # Crop the image with Numpy Array
            cropped = image_copy[row_from: row_to, col_from: col_to]
            cropped_images.append(cropped)  # add to the list
    return cropped_images


def save_image(image, name_prefix='untitled'):
    # make file name with the datetime suffix.
    d_date = datetime.datetime.now()  # get current datetime
    current_datetime = d_date.strftime("%Y%m%d%I%M%S")  # datetime to string
    file_path = name_prefix + '_'+ current_datetime + ".jpg"  # complete file name
    cv2.imwrite(file_path, image)
    return file_path

def process_image(image_file):
    image_origin = open_original(image_file)
    # image_origin = cv2.pyrUp(image_origin)  # size up ( x4 )  이미지 크기가 작을 경우 이미지 사이즈업 해야합니다.
    # Grey-Scale
    image_gray = get_gray(image_origin)
    cv2.imshow('image_gray', image_gray)
    # Morph Gradient
    image_gradient = get_gradient(image_gray)
    cv2.imshow('image_gradient', image_gradient)
    # Threshold
    image_threshold = get_threshold(image_gradient)
    cv2.imshow('image_threshold', image_threshold)
    # Long line remove
    image_line_removed = remove_long_line(image_threshold)
    cv2.imshow('image_line_removed', image_line_removed)
    # Morph Close
    image_close = get_closing(image_line_removed)
    contours = get_contours(image_close)
    cv2.imshow('image_close', image_close)

    return get_cropped_images(image_origin, contours)  # 글자로 추정되는 부분을 잘라낸 이미지들을 반환

read_configs('../test_images/screenshot/config_screenshot.yml')
image_path = '../test_images/screenshot/kimjunho.jpg'
cropped_images = process_image(image_path)  # process the image and get cropped screenshot
count = 0
hanbat_file_paths = ''
for crop_image in cropped_images:
    count += 1
    file_path = save_image(crop_image, "Hanbat_" + str(count))
    if (count == 10):
        hanbat_file_paths = file_path

img = cv2.imread(hanbat_file_paths)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hImg, wImg, _ = img.shape
print("-----------------------------------")
print(ocr.image_to_string(img, lang="kor"))
print("-----------------------------------")
boxes = ocr.image_to_boxes(img)
for b in boxes.splitlines():
    b= b.split(' ')
    x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, hImg-y), (w,hImg-h), (0,0,255), 1)
cv2.imshow('Hanbat', img)
cv2.waitKey(0)