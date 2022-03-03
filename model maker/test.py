import evaluate_model as ODT    # ODT -> Object Detection

model_path = './models/coin-best.tflite'

INPUT_IMAGE_URL = "./images/yen10.jpg"
DETECTION_THRESHOLD = 0.5 


ODT.main(model_path, INPUT_IMAGE_URL, DETECTION_THRESHOLD)
