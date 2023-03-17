import cv2
import pytesseract
import fitz
import numpy as np
import math

# added 2
# Load the PDF file
doc = fitz.open("input.pdf")

# Initialize a list to store the found objects and their corresponding text
found_objects = []

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
custom_config = r"--oem 3 --psm 6"

# Set up YOLOv3 neural network for object detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loop through each page of the PDF
for page in doc:
    # Extract the text from the page
    text = page.get_text()
    
    # Convert the PDF page to an image
    mat = np.array(page.getPixmap(matrix=fitz.Matrix(300/72,300/72)).samples)
    image = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

    # Use Tesseract OCR to extract text from the image
    text_boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
    
    # Use YOLOv3 to detect objects in the image
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Loop through each detected object
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Find the nearest text to the detected object
                nearest_text = ""
                nearest_distance = math.inf
                for i, text_box in enumerate(text_boxes["text"]):
                    distance = math.sqrt((center_x - text_boxes["left"][i])**2 + (center_y - text_boxes["top"][i])**2)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_text = text_box
                
                # Add the found object and its corresponding text to the list
                found_objects.append((classes[class_id], nearest_text))
    
# Print the list of found objects and their corresponding text
print(found_objects)
