from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import cv2
import pickle
import time

tic = time.time()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

pickle_id = open("ids.pickle","rb")
identity = pickle.load(pickle_id)

pickle_db = open("db.pickle","rb")
db_embed = pickle.load(pickle_db)

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

tac = time.time()

print(f'Files loaded in {tac - tic} seconds')

def is_match(capture, db, thresh=0.5):

    min_score = 1
    match = None
    for id in range(db.shape[0]):
        score = cosine(capture, db[id])
        if score <= min_score:
            match = identity[id]
            min_score = score
    if min_score <= thresh:
        print(f'Hi {match}! Score = {min_score:.2f}')
    else:
        print(f'No match found! Closest Match is {match}! Score = {min_score:.2f}')
    return False


def run():
    cap = cv2.VideoCapture(0)

    num_frames = 0

    status = True
    
    while status:
    # Read the frame
        _, img = cap.read()
    # Convert to grayscale
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        # Draw the rectangle around each face
        if len(faces) == 1:
            num_frames +=1
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display
        #cv2.imshow('Face Detector', img)
        if num_frames == 20:
            ROI = img[y:y+h, x:x+w]
            ROI = cv2.resize(ROI, (224,224))
            print(f'Face recognised in 20 frames')
            #cv2.imshow('Recognised', ROI)
            sample = asarray(ROI, 'float32')
            sample = np.reshape(sample,(1,224,224,3))
            sample = preprocess_input(sample, version=2)
            capture = model.predict(sample)
            print('Checking for Matches')
            status = is_match(capture, db_embed, thresh=0.5)
        # Stop if escape key is pressed
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
