from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import pickle
import cv2
import glob2


def get_array(file):    
    pixels = cv2.imread(file)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    (x,y,w,h) = faces[0]
    cv2.rectangle(pixels, (x, y), (x+w, y+h), (255, 0, 0), 2)
    ROI = pixels[y:y+h, x:x+w]
    ROI = cv2.resize(ROI, (224,224))
    facearray = asarray(ROI)
    return facearray



def main():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    filenames = glob2.glob("Image DS/*.jpeg")
    identity = [f[9:-5] for f in filenames]
    db_face = [get_array(f) for f in filenames]
    
    samples = asarray(db_face, 'float32')
    samples = preprocess_input(samples, version=2)
    db_embed = model.predict(samples)

    pickle_out = open("db.pickle","wb")
    pickle.dump(db_embed, pickle_out)
    pickle_out.close()

    pickle_out = open("ids.pickle","wb")
    pickle.dump(identity, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
		main()






