
""" You will write a python program that is capable of object detection from a .pdf file. The file input.pdf (version<1.5 ) contains multiple pages. In each page, there is a lot of objects  (either vector or bitmap images). Thus, the program will convert input.pdf into multiple bitmap images, one per each page of the .pdf document and put it into a folder "input_images" named as "input1.jpg", "input2.jpg", ... The resolution of these images will be higher then FullHD. Next, the program will load a pretrained EfficientDet D4 (1024x1024 input resolution) model (trained on COCO dataset) and do the object detection on the .jpg images. The model weights will be stored locally in the working directory. The program will have to resize the images to fit the model. The program will create a folder "output_bounding_boxes" which will contain the modified .jpg files but with bounding boxes indicating the detected objects. The files in the folder "output_bounding_boxes" will be named output1.jpg", "output2.jpg", ... The program will also print the classess of the detected objects per each page of the original document (per .jpg file) and also print  probability and the coordinates of the bounding box. The informations on where to get (download) all the necessary files to use the model EfficientDet D4  will be included in the program as comments.
 """


# This program will download the EfficientDet D4 model, convert the PDF file to images, run object detection on the images, 
# and save the output images with bounding boxes. 
# Note that the model file is large and may take some time to download. 
# Also, make sure the input PDF file is named input.pdf and is located in the same directory as the script.
# pip install pdf2image tensorflow opencv-python opencv-python-headless Pillow



import os
import sys
import tempfile
import requests
from pdf2image import convert_from_path
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import pickle



# Download EfficientDet D4 model
def download_model():
    model_url = "https://tfhub.dev/tensorflow/efficientdet/d4/1?tf-hub-format=compressed"
    response = requests.get(model_url)
    with open("efficientdet_d4.tar.gz", "wb") as f:
        f.write(response.content)
    print("Model downloaded")
    return "efficientdet_d4.tar.gz"


def load_model():
    model_file = "./efficientdet_d4"
    model = tf.saved_model.load(model_file)

    label_map_path = "./efficientdet_d4/mscoco_label_map.pbtxt"

    # Create an empty dictionary to store the class names
    class_names_dict = {}

    # Read the label map file and parse each class definition
    with open(label_map_path, 'r') as f:
        for line in f:
            if 'name:' in line:
                # Extract the class name from the line
                class_name = line.split('"')[1]
                # Extract the class ID from the previous line
                class_id = int(next(f).split(':')[-1])
                # Extract the display name from the next line
                display_name = next(f).split('"')[1]

                # Store the class name and ID in the dictionary
                class_names_dict[class_id] = display_name
    model.class_names_dict = class_names_dict
    # ids are not line indexes! this can not be used!
    #    # Load the class names from the file
    # with open('coco.names', "r") as f:
    #     class_names = [line.strip() for line in f]

    # # Set the class names dictionary as an attribute of the loaded model with dict
    # model.class_names_dict = {i: name for i, name in enumerate(class_names)}
    return model


def convert_pdf_to_images(input_pdf, output_folder):
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(input_pdf, output_folder=temp_dir, fmt="jpeg",poppler_path=r'C:\Program Files\poppler-0.68.0\bin') 
      
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for i, image in enumerate(images):
            image.save(os.path.join(output_folder, f"input{i + 1}.jpg"), "JPEG")
    return [os.path.join(output_folder, f"input{i + 1}.jpg") for i in range(len(images))]


def run_object_detection(model, image_path, output_folder):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (1024, 1024))

    input_tensor = tf.convert_to_tensor(img_resized, dtype=tf.uint8)
    input_tensor = tf.expand_dims(input_tensor, axis = 0)

    detections = model(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections

    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    image_with_boxes = img.copy()

    for i in range(num_detections):
        class_id = detections["detection_classes"][i]
        class_name = model.class_names_dict[class_id]
        score = detections["detection_scores"][i]
        bbox = detections["detection_boxes"][i]

        if score > 0.5:
            y1, x1, y2, x2 = bbox
            x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Detected object {class_name} with probability {score:.2f} at coordinates ({x1}, {y1}), ({x2}, {y2})")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_image_path = os.path.join(output_folder, "output" + os.path.basename(image_path)[5:])
    cv2.imwrite(output_image_path, image_with_boxes)

def scan_input_images():
    image_paths = []

    # List all files in the directory
    for file_name in os.listdir("input_images"):
        # Check if the file has a .jpg extension
        if file_name.endswith(".jpg"):
            # Get the relative path of the file
            rel_path = os.path.join("input_images", file_name)
            # Append the relative path to the list of image paths
            image_paths.append(rel_path)
            print(rel_path)

    return image_paths

def main():
    input_pdf = "input.pdf"
    input_images_folder = "input_images"
    output_bounding_boxes_folder = "output_bounding_boxes"

    #model_file = download_model()
    model = load_model()
    print('model loaded')


    #image_paths = convert_pdf_to_images(input_pdf, input_images_folder)
    image_paths = scan_input_images()

    for image_path in image_paths:
        print(f"Processing {image_path}")
        run_object_detection(model, image_path, output_bounding_boxes_folder)


if __name__ == "__main__":
    main()
