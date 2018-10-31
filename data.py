import numpy as np
import matplotlib.pyplot as plt
import mnist


data = mnist.MNIST('D:\programmeren\python\Machine learning\data')
data.gz = True
images,labels = data.load_training()

images = np.array(images)
labels = np.array(labels)


def c_img(img):
    return img.reshape((784,1))/255.0

def c_lbl(lbl):
    out = np.zeros((10,1))
    out[lbl-1] = 1
    return out


out_images = []
out_labels = []
for img,lbl in zip(images,labels):
    out_images.append(c_img(np.array(img)))
    out_labels.append(c_lbl(lbl))

images = out_images
labels = out_labels

images = np.array(out_images)
labels = np.array(out_labels)


