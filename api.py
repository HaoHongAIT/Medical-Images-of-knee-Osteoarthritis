# from tensorflow import keras
# from tensorflow_serving.apis import model_service_pb2_grpc
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2_grpc

# model = keras.models.load_model('my_model.h5')

# # Create a gRPC client to connect to the TensorFlow Serving server
# channel = grpc.insecure_channel('localhost:8500')
# stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# # Create a request to export the model
# request = model_service_pb2.ExportModelRequest()
# request.model_spec.name = 'my_model'
# request.model_spec.signature_name = 'serving_default'
# request.export_dir = '/path/to/export/directory'

# # Send the request to the TensorFlow Serving server
# response = stub.Export(request)


import cv2 as cv
import numpy as np
import tensorflow as tf
import os

class Pipeline:
    def __init__(self):
        self.img_path = None
        self.knee_xray = None
        self.model = tf.keras.models.load_model("./src/model/model_InceptionV3_DenseNet201_weights.h5")
    
    def pre_processing(self, img_name):
        self.img_path = f"./static/images/{img_name}"
        self.knee_xray = cv.imread(self.img_path)
        # resize image
        self.knee_xray = cv.resize(self.knee_xray, (224, 224))
        # rescale image
        self.knee_xray = np.array(self.knee_xray)
        self.knee_xray = self.knee_xray.astype("float")/255.0
        # expand dim
        self.knee_xray = np.expand_dims(self.knee_xray, axis=0) 
        
    def predict(self,):
        result = self.model.predict(self.knee_xray)
        level_dict = {0: "Normal", 1:"mức độ 1", 2:"mức độ 2", 3:"mức độ", 4:"mức độ 4"}        
        level = np.argmax(result)
        params={
                'image': self.img_path,
                'predict': "có bệnh" if level>0 else "không có bệnh",
                'level' : level_dict[level]
            }
        return params
    
# if __name__ == '__main__':        
#     model = dl_model(img_path="./static/images/9003175L.png")
#     model.pre_processing()       
#     print(model.predict())