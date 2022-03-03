import obj_detector as ODT
import config
import cv2
import numpy as np


def get_params():
    params = dict()
    params["model_path"] = './models/coin-best.tflite'
    params["img_path"] = "./images/yen10.jpg"
    params["detect_threshold"] = 0.5
    params["frame_path"] = "./images/frame.jpg"
    return params

if __name__=="__main__":
    params = get_params()
    ODT.main(params["model_path"], params["img_path"], params["detect_threshold"])