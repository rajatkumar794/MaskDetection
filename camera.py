import cv2
from model import MaskDetectionModel
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model

detector=MTCNN()
model = load_model('model.h5')
labels = ['balaclava_ski_mask', 'eyeglasses', 'face_no_mask',
       'face_other_covering', 'face_shield', 'face_with_mask',
       'face_with_mask_incorrect', 'gas_mask', 'goggles', 'hair_net',
       'hat', 'helmet', 'hijab_niqab', 'hood', 'mask_colorful',
       'mask_surgical', 'other', 'scarf_bandana', 'sunglasses', 'turban']
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        #print(fr)
        fr = cv2.flip(fr, 1)
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(fr)

        for face in faces:
            bounding_box = face['box']
            inp = gray_fr[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
            try:
                inp=cv2.resize(inp,(50,50))
            except:
                break
            inp=inp.reshape(-1,50,50,1)
            pred=model.predict(inp.astype('float32'))
            pred=labels[np.argmax(pred)]
            #fr = cv2.resize(fr, (640, 480))
            cv2.putText(fr, pred, (bounding_box[0], bounding_box[1]), font, 1, (0, 0, 255), 2)
            cv2.rectangle(fr,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (255,0,0),2)
            fr=cv2.resize(fr,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
