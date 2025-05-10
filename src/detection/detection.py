"""A module for classifying directional arrows using TensorFlow."""

import cv2
import tensorflow as tf
import time
import numpy as np
import uuid
from src.common import utils
import os

POINT_THRESHOLD_NUM_PIXELS = 5
INTERSECTION_THRESHOLD_NUM_PIXELS = 15
ARROW_MIN_ANGLE = 70
ARROW_MAX_ANGLE = 110

DEBUG = False
TF = False

#########################
#       Functions       #
#########################
def load_model():
    """
    Loads the saved model's weights into an Tensorflow model.
    :return:    The Tensorflow model object.
    """
    if TF:
        model_dir = f'assets/models/rune_model_rnn_filtered_cannied/saved_model'
        return tf.saved_model.load(model_dir)
    return None


def canny(image):
    """
    Performs Canny edge detection on IMAGE.
    :param image:   The input image as a Numpy array.
    :return:        The edges in IMAGE.
    """

    image = cv2.Canny(image, 200, 300)
    colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return colored


def filter_color(image):
    """
    Filters out all colors not between orange and green on the HSV scale, which
    eliminates some noise around the arrows.
    https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
    :param image:   The input image.
    :return:        The color-filtered image.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv, (1, 75, 100), (85, 255, 255))
    mask_2 = cv2.inRange(hsv, (95, 75, 100), (180, 255, 255))
    mask = cv2.bitwise_or(mask_1, mask_2)

    # Mask the image
    color_mask = mask > 0
    arrows = np.zeros_like(image, np.uint8)
    arrows[color_mask] = image[color_mask]
    return arrows


def run_inference_for_single_image(model, image):
    """
    Performs an inference once.
    :param model:   The model object to use.
    :param image:   The input image.
    :return:        The model's predictions including bounding boxes and classes.
    """

    image = np.asarray(image)

    with tf.device('/GPU:0'):
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis,...]

        model_fn = model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0,:num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        return output_dict


def sort_by_confidence(model, image):
    """
    Runs a single inference on the image and returns the best four classifications.
    :param model:   The model object to use.
    :param image:   The input image.
    :return:        The model's top four predictions.
    """

    output_dict = run_inference_for_single_image(model, image)
    zipped = list(zip(output_dict['detection_scores'],
                      output_dict['detection_boxes'],
                      output_dict['detection_classes']))
    pruned = [t for t in zipped if t[0] > 0.5]
    pruned.sort(key=lambda x: x[0], reverse=True)
    result = pruned[:4]
    return result


def get_boxes(model, image):
    """
    Returns the bounding boxes of the top four classified arrows.
    :param model:   The model object to predict with.
    :param image:   The input image.
    :return:        Up to four bounding boxes.
    """

    output_dict = run_inference_for_single_image(model, image)
    zipped = list(zip(output_dict['detection_scores'],
                      output_dict['detection_boxes'],
                      output_dict['detection_classes']))
    pruned = [t for t in zipped if t[0] > 0.5]
    pruned.sort(key=lambda x: x[0], reverse=True)
    pruned = pruned[:4]
    boxes = [t[1:] for t in pruned]
    return boxes


# @utils.run_if_enabled
def merge_detection(model, image):
    """
    Run two inferences: one on the upright image, and one on the image rotated 90 degrees.
    Only considers vertical arrows and merges the results of the two inferences together.
    (Vertical arrows in the rotated image are actually horizontal arrows).
    :param model:   The model object to use.
    :param image:   The input image.
    :return:        A list of four arrow directions.
    """
    label_map = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
    converter = {'up': 'right', 'down': 'left'}         # For the 'rotated inferences'
    classes = []
    
    # Preprocessing
    height, width, channels = image.shape
    cropped = image[23 * height // 100:40 * height // 100, 32 * width // 100:66 * width // 100]
    filtered = filter_color(cropped)
    cannied = canny(filtered)
    uuid_1 = uuid.uuid1()
    # cv2.imwrite(f"assets/training/{uuid_1}-raw.png", filtered)
    # cv2.imwrite(f"assets/training/{uuid_1}-filtered.png", filtered)
    # cv2.imwrite(f"assets/training/{uuid_1}-cannied.png", cannied)

    # Isolate the rune box
    height, width, channels = cannied.shape
    boxes = get_boxes(model, cannied)
    if len(boxes) == 4:      # Only run further inferences if arrows have been correctly detected
        y_mins = [b[0][0] for b in boxes]
        x_mins = [b[0][1] for b in boxes]
        y_maxes = [b[0][2] for b in boxes]
        x_maxes = [b[0][3] for b in boxes]
        left = int(round(min(x_mins) * width))
        right = int(round(max(x_maxes) * width))
        top = int(round(min(y_mins) * height))
        bottom = int(round(max(y_maxes) * height))
        rune_box = cannied[top:bottom, left:right]

        # Pad the rune box with black borders, effectively eliminating the noise around it
        height, width, channels = rune_box.shape
        pad_height, pad_width = 384, 455
        preprocessed = np.full((pad_height, pad_width, channels), (0, 0, 0), dtype=np.uint8)
        x_offset = (pad_width - width) // 2
        y_offset = (pad_height - height) // 2

        if x_offset > 0 and y_offset > 0:
            preprocessed[y_offset:y_offset+height, x_offset:x_offset+width] = rune_box

        # Run detection on preprocessed image
        lst = sort_by_confidence(model, preprocessed)
        lst.sort(key=lambda x: x[1][1])
        classes = [label_map[item[2]] for item in lst]

        # Run detection on rotated image
        rotated = cv2.rotate(preprocessed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        lst = sort_by_confidence(model, rotated)
        lst.sort(key=lambda x: x[1][2], reverse=True)
        rotated_classes = [converter[label_map[item[2]]]
                           for item in lst
                           if item[2] in [1, 2]]
            
        # Merge the two detection results
        for i in range(len(classes)):
            if rotated_classes and classes[i] in ['left', 'right']:
                classes[i] = rotated_classes.pop(0)

        if len(classes) != 4:
            cv2.imwrite(f"assets/training/{uuid_1}.png", preprocessed)
            cv2.imwrite(f"assets/training/{uuid_1}-rotated.png", rotated)
    return classes

# @utils.run_if_enabled
def find_arrows(image):
    processed_image = preprocess(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(filter_contour, contours)

    polygons = []
    for contour in contours:
        # lower arclength_filter number means tighter approximation
        arclength_filter = 0.01 * cv2.arcLength(contour, True)
        polygons.append(cv2.approxPolyDP(contour, arclength_filter,True))

    if DEBUG:
        cv2.drawContours(image, polygons, -1, (0, 255, 0), 3)
        cv2.imshow("arrows", processed_image)

    arrows = []
    # Get polygons
    for polygon in polygons:
        n_points = len(polygon)
        if n_points < 3:
            continue
        sides = []
        for i, vertex in enumerate(polygon):
            p1 = vertex[0]
            p2 = polygon[(i + 1) % n_points][0]
            d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            if 10 <= d <= 20:
                sides.append((p1, p2, d))
        # sort sides by length
        sides = sorted(sides, key=lambda line: (line[2], line[0][0]), reverse=True)[:2]

        # Solve arrows and sort by x-axis
        debug_print_lines(sides, processed_image, (255, 0, 0))
        arrow = solve_arrow_horizontal(sides)
        if arrow is not None:
            if DEBUG:
                debug_print_lines(sides, processed_image, (50, 50, 0))
                print(arrow)
            arrows.append(arrow)
        else:
            arrow = solve_arrow_vertical(sides)
            if arrow is not None:
                if DEBUG:
                    debug_print_lines(sides, processed_image, (50, 50, 0))
                    print(arrow)
                arrows.append(arrow)

    # uuid_1 = uuid.uuid1()
    # cv2.imwrite(f"assets/training/{uuid_1}-raw.png", image)
    # cv2.imwrite(f"assets/training/{uuid_1}-processed.png", processed_image)

    num_arrows = len(arrows)
    if num_arrows < 4:
        return []

    sorted_arrows = sorted(arrows, key=lambda arrow: arrow[1])
    results = [sorted_arrows[0]]
    for i in range(1, num_arrows):
        if abs(sorted_arrows[i][1] - sorted_arrows[i - 1][1]) >= POINT_THRESHOLD_NUM_PIXELS:
            results.append(sorted_arrows[i])
        if len(results) == 4:
            return get_arrows(results)

    # cv2.imshow("arrows", image)
    # cv2.waitKey(3000)
    return get_arrows(results)

def debug_print_lines(sides, image, color):
    if DEBUG:
        count = 0
        for side in sides:
            cv2.line(image, side[0], side[1], color, 2)
            cv2.imshow("arrows", image)
            cv2.waitKey(3000)
            count += 1
            if count > 1:
                break

def preprocess(image):
    height, width, channels = image.shape
    cropped = image[23 * height // 100:40 * height // 100, 32 * width // 100:66 * width // 100]
    if DEBUG:
        cropped = image

    hsv = cv2.cvtColor(filter_color(cropped), cv2.COLOR_RGB2HSV)
    black_white = hsv[:, :, 0]
    blurred = cv2.GaussianBlur(black_white, (5, 5), 1)
    blurred = cv2.medianBlur(blurred, 5)
    # cv2.imshow("processed", blurred)
    cannied = cv2.Canny(blurred, 50, 50)

    return cannied

def filter_contour(contour):
    return 300 <= cv2.contourArea(contour) <= 800

def solve_arrow_horizontal(sides):
    if len(sides) != 2:
        return None

    # sort by y-axis, if points are too close together then it's not horizontal
    points = get_points(sides)
    sorted_points = sorted(points, key=lambda point: point[1])

    direction, x_coordinate = solve_arrow(sorted_points, True)
    if direction == 0:
        return None

    return ('left', x_coordinate) if direction == -1 else ('right', x_coordinate)

def solve_arrow_vertical(sides):
    if len(sides) != 2:
        return None

    # sort by x-axis, if points are too close together then it's not vertical
    points = get_points(sides)
    sorted_points = sorted(points, key=lambda point: point[0])

    direction, x_coordinate = solve_arrow(sorted_points, False)
    if direction == 0:
        return None

    return ('up', x_coordinate) if direction == -1 else ('down', x_coordinate)

def get_points(sides):
    points = []
    for side in sides:
        points.append((side[0][0], side[0][1]))
        points.append((side[1][0], side[1][1]))

    return points

def solve_arrow(sorted_points, is_horizontal):
    # Remove intersecting point
    sorted_points = remove_intersecting_point(sorted_points)
    if len(sorted_points) != 3:
        return 0, 0

    x1, y1 = sorted_points[0]
    x2, y2 = sorted_points[1]
    x3, y3 = sorted_points[2]
    if not is_horizontal:
        direction = check_coordinates(x1, x2, x3, y1, y2, y3)
    else:
        direction = check_coordinates(y1, y2, y3, x1, x2, x3)

    if direction == 0:
        return direction, 0

    angle = get_angle_between_points(
        np.array([x1, y1]),
        np.array([x3, y3]),
        np.array([x2, y2])
    )

    if DEBUG:
        print(f"angle {angle}")

    if angle < ARROW_MIN_ANGLE or angle > ARROW_MAX_ANGLE:
        return 0, 0

    return direction, x2

def remove_intersecting_point(points):
    if len(points) != 4:
        return points

    # Remove intersecting point
    if (abs(points[1][0] - points[2][0]) < INTERSECTION_THRESHOLD_NUM_PIXELS and
        abs(points[1][1] - points[2][1]) < INTERSECTION_THRESHOLD_NUM_PIXELS):
        points.pop(1)
        return points
    else:
        return points

def check_coordinates(a1, a2, a3, b1, b2, b3):
    # Check if points on axis too close
    if (abs(a1 - a2) < POINT_THRESHOLD_NUM_PIXELS or
        abs(a3 - a1) < POINT_THRESHOLD_NUM_PIXELS or
        abs(a2 - a3) < POINT_THRESHOLD_NUM_PIXELS):
        return 0

    # Check if vertex is lower or higher than points
    if b2 < b1 and b2 < b3:
        return -1

    if b2 > b1 and b2 > b3:
        return 1

    return 0

def get_angle_between_points(p1, p2, vertex):
    v1 = get_unit_vector(p1 - vertex)
    v2 = get_unit_vector(p2 - vertex)
    return get_angle(v1, v2)

def get_unit_vector(v):
    return v / np.linalg.norm(v)

def get_angle(v1, v2):
    return np.rad2deg(
        np.arccos(np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1
        ))
    )

def get_arrows(arrows):
    result = []
    if len(arrows) != 4:
        return result

    for arrow in arrows:
        result.append(arrow[0])
    return result

# Script for testing the detection module by itself
if __name__ == '__main__':
    from src.common import config, utils
    import mss
    config.enabled = True
    monitor = {'top': 0, 'left': 0, 'width': 1366, 'height': 768}
    # model = load_model()
    while True:
        with mss.mss() as sct:
            frame = np.array(sct.grab(monitor))
            #cv2.imshow('frame', canny(filter_color(frame)))
            # arrows = merge_detection(model, frame)
            arrows = find_arrows(frame)
            print(arrows)
            if cv2.waitKey(10000) & 0xFF == 27:     # 27 is ASCII for the Esc key
                break
