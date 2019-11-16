import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '_data/pose_deploy.prototxt'
PRETRAINED = '_data/pose_iter_102000.caffemodel'

# load the model
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Classifier(MODEL_FILE, PRETRAINED)#, mean=np.load('data/train_mean.npy').mean(1).mean(1), channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))
print("successfully loaded classifier")

# test on a image
IMAGE_FILE = '.\HandKeyPointDetector-master\data\hand.jpg'
input_image = caffe.io.load_image(IMAGE_FILE)
# predict takes any number of images,
# and formats them for the Caffe net automatically
pred = net.predict([input_image])

print(pred)

