import data
import numpy as np
import matplotlib.pyplot as plt
import time


def sig(x):
    return 1/(1+np.exp(-x))

def d_sig(x):
    return sig(x)*(1-sig(x))


class Network:

    def __init__(self,shape=(),data=(None,None),funct=sig,d_funct=d_sig):
        self.f = funct
        self.d_f = d_funct

        self.length = len(shape)
        self.shape = shape

        self.all_x = np.array([np.zeros((i,1)) for i in shape[:]])
        self.all_dz = np.array([np.zeros((i,1)) for i in shape[1:]])

        self.w_grad = np.array([np.zeros((i,j)) for i,j in zip(shape[1:],shape)])
        self.b_grad = np.array([np.zeros((i,1)) for i in shape[1:]])

        self.xs = np.array(data[0])
        self.ys = np.array(data[1])

        self.ws = np.array([np.zeros((i,j)) for i,j in zip(shape[1:],shape)])
        self.bs = np.array([np.zeros((i,1)) for i in shape[1:]])


    def cost(self,xs,ys):
        return np.sum((self.neur_sum(xs)-ys)**2)/(len(xs)*len(ys[0]))


    def neur_sum(self,x):

        for w,b in zip(self.ws,self.bs):
            x = self.f(w@x+b)

        return x


    def gradient(self,x,y):

        L = 0
        self.all_x[L] = x
        for w,b in zip(self.ws,self.bs):
            z = w@x+b
            self.all_dz[L] = self.d_f(z)
            x = self.f(z)
            self.all_x[L+1] = x
            L += 1


        dc_dx = x-y
        
        L = self.length-2
        while L >= 0:
            dc_db = dc_dx*self.all_dz[L]
            dc_dw = dc_db@self.all_x[L].T
            dc_dx = self.ws[L].T@(dc_dx*self.all_dz[L])
            self.w_grad[L] = dc_dw
            self.b_grad[L] = dc_db
            L -= 1

        return self.w_grad,self.b_grad



test = Network(shape=(784,15,10),data=(data.images,data.labels))

print(test.cost(np.array(data.images[:1000]),np.array(data.labels[:1000])))

for i in range(1000):
    for x,y in zip(np.array(data.images[:1000]),np.array(data.labels[:1000])):
        w,b = test.gradient(x,y)
        test.ws -= w
        test.bs -= b


