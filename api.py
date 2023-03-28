import cv2 as cv
import numpy as np
import tensorflow as tf
import os

class Pipeline:
    def __init__(self, img_name):
        self.img_path = f"./static/images/{img_name}"
        self.knee_xray = None
        self.result = None
    
    def detect_to_crop(self):
        pass
    
    def pre_processing(self):
        self.knee_xray = cv.imread(self.img_path)
        # resize image
        self.knee_xray = cv.resize(self.knee_xray, (224, 224))
        # rescale image
        self.knee_xray = np.array(self.knee_xray)
        self.knee_xray = self.knee_xray.astype("float")/255.0
        # expand dim
        self.knee_xray = np.expand_dims(self.knee_xray, axis=0) 
        
    def predict(self, model):
        result = model.predict(self.knee_xray)
        level_dict = {0: "Normal", 1:"mức độ 1", 2:"mức độ 2", 3:"mức độ", 4:"mức độ 4"}        
        level = np.argmax(result)
        self.result = {
                'image': self.img_path,
                'predict': "có bệnh" if level>0 else "không có bệnh",
                'level' : level_dict[level],
            }
        return self.result
