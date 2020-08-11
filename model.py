from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class MaskDetectionModel(object):

    labels = ['eyeglasses', 'face_no_mask', 'face_with_mask', 'helmet', 'hood']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
    	with session.as_default():
    		with session.graph.as_default():
    			self.preds = self.loaded_model.predict(img)
    			return MaskDetectionModel.labels[np.argmax(self.preds)]