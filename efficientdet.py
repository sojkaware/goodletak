import cv2
import numpy as np
import tensorflow as tf

# Load the model from the saved weights file
model = tf.keras.models.load_model('efficientdet-d4.h5', compile=False)

# Load the class labels from a file
with open('coco.names', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Load the input image
image = cv2.imread('image.jpg')

# Preprocess the input image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image, (512, 512))
image_normalized = image_resized / 255.0
input_data = np.expand_dims(image_normalized, axis=0)

# Perform inference with the model
outputs = model.predict(input_data)

# Extract the bounding boxes, class IDs, and scores
boxes = outputs[0][:, :4]
class_ids = np.argmax(outputs[0][:, 4:], axis=-1) + 1
scores = np.max(outputs[0][:, 4:], axis=-1)

# Filter out weak detections
detections = []
for box, class_id, score in zip(boxes, class_ids, scores):
    if score > 0.5:
        detections.append((box, class_id, score))

# Draw the bounding boxes and labels on the input image
for box, class_id, score in detections:
    x_min, y_min, x_max, y_max = box
    x_min = int(x_min * image.shape[1])
    y_min = int(y_min * image.shape[0])
    x_max = int(x_max * image.shape[1])
    y_max = int(y_max * image.shape[0])
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    label = class_labels[class_id - 1] + ': {:.2f}'.format(score)
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Display the output image
cv2.imshow('Output', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
