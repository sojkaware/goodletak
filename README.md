# goodletak
# notes
IBM Watson Visual Recognition provides up to 10,000 API calls per month with a maximum image size of 10 MB per image for free.
Google Cloud Vision API provides up to 1,000 API requests per month for free, with a maximum image size of 5 MB per image. 
others
Microsoft Azure Cognitive Services
Amazon Rekognition:
Clarifai


MobileNetV2 network 1000 categories, 15MB weights,  classify mostly
YOLOv3 network 80 categories, 250MB, detect mostly and then classify

 Na loading


 import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib.request

# Download the yolov3.weights file
url_weights = "https://pjreddie.com/media/files/yolov3.weights"
filename_weights = "yolov3.weights"
urllib.request.urlretrieve(url_weights, filename_weights)

# Download the yolov3.cfg file
url_cfg = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
filename_cfg = "yolov3.cfg"
urllib.request.urlretrieve(url_cfg, filename_cfg)


# Conda tips

conda create --name my_project_env python=3.8
conda activate my_project_env
conda info --envs
conda list

kdzy mi to napise ze  nebyl nalezen v current channel
conda install -c conda-forge pymupdf
conda config --add channels conda-forge

to create new enviroment from requirements.txt
conda create --name my_new_project_env --file requirements.txt

udelat requirements
pip freeze > requirements.txt

jak udelat requirements z kodu - V pracovnim adreesari
pip install pipreqs
pipreqs .



You will write a python program that is capable of object detection from a .pdf file. The file input.pdf (version<1.5 ) contains multiple pages. In each page, there is a lot of objects  (either vector or bitmap images). Thus, the program will convert input.pdf into multiple bitmap images, one per each page of the .pdf document and put it into a folder "input_converted" named as "input1.jpg", "input2.jpg", ... The resolution of these images will be higher then FullHD. Next, the program will load a pretrained EfficientDet D4 (1024x1024 input resolution) model (trained on COCO dataset) and do the object detection on the .jpg images. The model weights will be stored locally in the working directory. The program will have to resize the images to fit the model. The program will create a folder "output_bounding_boxes" which will contain the modified .jpg files but with bounding boxes indicating the detected objects. The files in the folder "output_bounding_boxes" will be named output1.jpg", "output2.jpg", ... The program will also print the classess of the detected objects per each page of the original document (per .jpg file) and also print  probability and the coordinates of the bounding box. The informations on where to get (download) all the necessary files to use the model EfficientDet D4  will be included in the program as comments.


directory comprehension
squares = {x: x*x for x in range(1, 6)}  		# prints: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
names = {i: name for i, name in enumerate(class_names)}	# tady mam ve foru 2

how to pickle - zavisle objekty ale picklovat nejdou
    # Save the variables to a file using pickle
    with open("variables.pickle", "wb") as f:
        pickle.dump((myvar), f)

    # Load the variables from the file using pickle
    with open("variables.pickle", "rb") as f:
        myvar = pickle.load(f)
