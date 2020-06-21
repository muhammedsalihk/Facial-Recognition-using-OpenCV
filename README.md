# Facial-Recognition-using-OpenCV

## Introduction
With the onset of deep learning CNN models, facial recognition systems have become prevalent and  are used widely nowadays. 

The fundamental principle that makes these systems work is the use of a neural network that is trained to produce an embedding for the image of person’s face such that similar images (i.e. the images of the same person) produces similar embeddings that are close in the vector space, and dissimilar images produces farther embeddings.

In this project, we build a facial recognition system using OpenCV to detect the faces from live webcam feed and then use embeddings from a pretrained keras model to do the matching process.


## Methodology
The images are captured from a live webcam feed on which the faces are detected and localised using the haar face cascades provided in the open CV package. To ensure stable images are captured, the system waits for atleast 20 frames in which it detects an face before the facial recognition process is initiated.

Once the frontal face image is obtained, it is passed through the network to obtain the embedding. This embedding is then checked against the embeddings of the images stored in the database. Based on a criterion for minimum similarity, the best match is reported.

![Flow1](https://github.com/muhammedsalihk/Facial-Recognition-using-OpenCV/blob/master/Images/Flow%201.png)

![Flow2](https://github.com/muhammedsalihk/Facial-Recognition-using-OpenCV/blob/master/Images/Flow2.png)

## Files

update_db.py – To generate the database embeddings for the images stored in the folder ‘Image DS’. Please note that the name of the image file is extracted as the person’s name. To add a new person to the database, save their image in the Image DS folder with filename {Person's Name}.jpeg and then run update_db.py

run.py – To run the facial recognition process. The match found is reported in the terminal. To customise the functionality or to use another camera instead of the default webcam, the file exec.py can be edited.
