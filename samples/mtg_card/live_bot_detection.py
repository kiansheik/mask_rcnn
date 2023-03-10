# %%
import json

import tensorflow.compat.v1 as tf

import mrcnn.model as modellib
from mrcnn.config import Config

tf.config.set_visible_devices([], "GPU")
import os

import cv2
import fuzzy_search
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en")
OBJ_CLASSES = ["card", "name", "set_symbol"]
BASE_SIZE = (1024, 1024)
ROOT_DIR = os.path.abspath("../../")


def get_file_paths(dir_path):
    file_paths = []
    for root, directories, files in os.walk(dir_path):
        for file in files:
            if not file.startswith("."):
                file_path = os.path.join(root, file)
                file_path = os.path.abspath(file_path)
                file_paths.append(file_path)
    return file_paths


class MTGCardInferConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "mtg_card"
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = len(OBJ_CLASSES) + 1  # Background + mtg_card classes
    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7
    USE_MINI_MASK = False
    IMAGE_MIN_DIM = BASE_SIZE[0]
    IMAGE_MAX_DIM = BASE_SIZE[1]
    BACKBONE = "resnet101"


data_dir = r"/Users/kiansheik/code/mydata/processed_pics"
logs_dir = os.path.join(os.path.join(ROOT_DIR, "assets", "augmented_data"), "logs")

config = MTGCardInferConfig()
inference_model = model = modellib.MaskRCNN(
    mode="inference", config=config, model_dir=logs_dir
)
weights_path = model.find_last()
print(weights_path)

tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True)

dataset_dir = os.path.join(data_dir, "test")
files = get_file_paths(dataset_dir)


# %%
def reverse_resize_and_pad(padded_image, original_size):
    # Get the current height and width of the padded image
    height, width = padded_image.shape[:2]
    # Get the current height and width of the image
    rows, cols = original_size[:2]

    # Calculate the scaling factor for the height and width
    height_scale = height / rows
    width_scale = width / cols
    scale = min(height_scale, width_scale)

    # Calculate the new height and width of the image
    new_height = int(rows * scale)
    new_width = int(cols * scale)

    # Calculate the padding needed
    top_pad = 0
    bottom_pad = 0
    left_pad = 0
    right_pad = 0
    if new_height < height:
        top_pad = (height - new_height) // 2
        bottom_pad = height - new_height - top_pad
    if new_width < width:
        left_pad = (width - new_width) // 2
        right_pad = width - new_width - left_pad

    # Crop the image to remove the padding
    cropped_image = padded_image[
        top_pad : height - bottom_pad, left_pad : width - right_pad
    ]
    # Resize the image
    resized_image = cv2.resize(cropped_image, (cols, rows))

    return resized_image


def resize_and_pad(image, target_shape):
    height, width = target_shape
    # Get the current height and width of the image
    rows, cols, channels = image.shape

    # Calculate the scaling factor for the height and width
    height_scale = height / rows
    width_scale = width / cols
    scale = min(height_scale, width_scale)

    # Calculate the new height and width of the image
    new_height = int(rows * scale)
    new_width = int(cols * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate the padding needed
    top_pad = 0
    bottom_pad = 0
    left_pad = 0
    right_pad = 0
    if new_height < height:
        top_pad = (height - new_height) // 2
        bottom_pad = height - new_height - top_pad
    if new_width < width:
        left_pad = (width - new_width) // 2
        right_pad = width - new_width - left_pad

    # Pad the image
    padded_image = cv2.copyMakeBorder(
        resized_image,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    return padded_image


def bitmask_to_bounding_box(bitmask):
    # Find the non-zero elements in the bitmask
    non_zero_elements = cv2.findNonZero(bitmask)
    # Use the minAreaRect function to find the bounding box of the rotated rectangle
    rect = cv2.minAreaRect(non_zero_elements)
    # Extract the bounding box points from the rect object
    bounding_box = cv2.boxPoints(rect)
    # Convert the bounding box points to integer values
    bounding_box = np.int0(bounding_box)
    return bounding_box


def get_best_match(image, card_name):
    image_list = fuzzy_search.get_set_images_by_name(
        card_name.lower(), shape=image.shape
    )
    query_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Back to 3 channel because I am having trouble with 1 channel
    query_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    # Convert the images to uint8
    query_image = cv2.normalize(
        query_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    reference_images = [
        np.uint8(cv2.cvtColor((255 - x[0][:, :, -1]), cv2.COLOR_GRAY2BGR))
        for x in image_list
    ]
    reference_images = [
        cv2.normalize(
            x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        for x in reference_images
    ]
    # Compute the distances
    print(query_image.shape, type(query_image), query_image.dtype)
    print(
        reference_images[0].shape, type(reference_images[0]), reference_images[0].dtype
    )
    # Create an ORB object
    orb = cv2.ORB_create()

    # Compute the keypoints and descriptors for the query image
    query_kp, query_des = orb.detectAndCompute(query_image, None)

    # Initialize the minimum distance and index
    min_distance = float("inf")
    min_index = -1

    # Loop over the reference images
    for i, reference_image in enumerate(reference_images):
        # Compute the keypoints and descriptors for the reference image
        reference_kp, reference_des = orb.detectAndCompute(reference_image, None)

        # Use the Brute-Force matcher to find the closest keypoints
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(query_des, reference_des)
        except Exception as e:
            print("FUCK")
            print(e)
            print(query_des, reference_des)
            return image_list[0]

        # Compute the total distance
        distance = sum([m.distance for m in matches])

        # Update the minimum distance and index if necessary
        if distance < min_distance:
            min_distance = distance
            min_index = i

    # Return the reference image with the minimum distance
    return image_list[min_index]


# %%
def draw_boxes(image, masks, class_ids, scores):
    best_guess = {k: (-1 * float("Inf"), None) for k in set(class_ids)}
    for i, class_id in enumerate(class_ids):
        top = best_guess[class_id]
        if top[0] < scores[i]:
            best_guess[class_id] = (scores[i], i)
    for class_id, (score, i) in best_guess.items():
        mask = masks[:, :, i]
        label = OBJ_CLASSES[class_id - 1]
        coords = bitmask_to_bounding_box(mask.astype(np.uint8))
        cv2.polylines(image, [coords], True, (0, 0, 255), 2)
    for class_id, (score, i) in best_guess.items():
        mask = masks[:, :, i]
        # class_id = class_ids[i]
        label = OBJ_CLASSES[class_id - 1]
        coords = bitmask_to_bounding_box(mask.astype(np.uint8))
        x, y, w, h = cv2.boundingRect(coords)
        center_x, center_y = x + w // 2, y + h // 2
        # Convert the coordinates to a numpy array
        coords = np.array(coords, dtype=np.int32)
        # Add black border
        cv2.putText(
            image,
            label,
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        # Put the label text on the image
        cv2.putText(
            image,
            label,
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    card_mask = masks[:, :, best_guess[OBJ_CLASSES.index("card") + 1][1]]
    card_coords = bitmask_to_bounding_box(card_mask.astype(np.uint8))
    # Get the center-point of the card
    x, y, w, h = cv2.boundingRect(card_coords)
    center = (x + w // 2, y + h // 2)

    # Find top left corner as it's the closest to the farthest corner
    return image, center


# %%
def detect_and_guess(model, image):
    # Run model detection and generate the color splash effect
    # print("Running on {}".format(image_path))
    # # Read image
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    orig_shape = image.shape[:2]
    resized_image = resize_and_pad(image, (1024, 1024))
    # Detect objects
    r = model.detect([resized_image], verbose=1)[0]
    resized_masks = []
    for i in range(r["masks"].shape[2]):
        mask = r["masks"][:, :, i]
        nm = reverse_resize_and_pad(mask.astype(np.uint8), orig_shape)
        resized_masks.append(nm)
    r_masks = np.stack(resized_masks, axis=2)
    return draw_boxes(image, r_masks, r["class_ids"], r["scores"])


# Open a connection to the webcam
cap = cv2.VideoCapture(0)


with open(
    "/Users/kiansheik/code/card-sorting-bot/grbl/calibration_images/calibration.json"
) as f:
    calib = json.load(f)
    camera_matrix = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["dist_coeff"])


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    # rotation_matrix = cv2.getRotationMatrix2D((frame.shape[0]/2, frame.shape[1]/2), 180, 1)
    # frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    frame = cv2.flip(frame, 1)  # flip the image vertically
    frame = cv2.transpose(frame)
    try:
        frame, target = detect_and_guess(model, frame)
        print(target)
    except:
        print("Nothing to print")

    # Show the frame with the bounding boxes and text content
    cv2.imshow("QR Code", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break
        # Break the loop if the 'q' key is pressed
    if cv2.waitKey(100) & 0xFF == ord("p"):
        print(target)

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
