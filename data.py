import numpy as np
import matplotlib.pyplot as plt
import mnist


data = mnist.MNIST('path of your data')
data.gz = True
images,labels = data.load_training()

images = np.array(images)
labels = np.array(labels)

def c_lbl(lbl):
    out = np.zeros((10,1))
    out[lbl-1] = 1
    return out


out_labels = []
for img,lbl in zip(images,labels):
    out_labels.append(c_lbl(lbl))


images = images.reshape(60000,784,1)/255.0
labels = np.array(out_labels)

all_data = []
for i,l in zip(images,labels):
    all_data.append((i,l))





