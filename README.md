# GWM-Scripts
This scripts are created by Shaurya. He holds the copyright of the same.

Draw annotation boxes: This is a IPython notebook which contains the code to draw annotation boxes in the images from the annotations present in csv file. You should specify the path of the images and the corresponding csv file.

Generate tfrecord: This file contains the code to generate tfrecords. You should specify the path of the images, corresponding csv file and the path to save tfrecord.

ReadingTfRecord: This file contains code to read the tfrecord as string which can also be used to validate the tfrecord.

VGG16: This file contains the code to load the pretrained vgg16 model trained on imagenet. It then converts the model and trains it to classify coco animal dataset by replacing the output layer to classify 8 number of classes..
