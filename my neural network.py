import numpy as np
import matplotlib.pyplot as plt


def sig(x):
    return 1/(1+np.exp(-x))

def d_sig(x):
    return x*(1-x)


class Network:

    def __init__(self,shape=(),data=(None,None)):
        self.lenght = len(shape)
        self.shape = shape

        self.xs = data[0]
        self.ys = data[1]

        self.ws = np.array([np.zeros((i,j)) for i,j in zip(shape[1:],shape)])
        self.bs = np.array([np.zeros((i,1)) for i in shape[1:]])

        self.w_grad = np.array([np.zeros((i,j)) for i,j in zip(shape[1:],shape)])
        self.b_grad = np.array([np.zeros((i,1)) for i in shape[1:]])

        self.tot_grad = np.array([[np.zeros((i,j)) for i,j in zip(shape[1:],shape)],[np.zeros((i,1)) for i in shape[1:]]])


    def neur_sum(self,x):

        for w,b in zip(self.ws,self.bs):
            x = sig(w@x+b)

        return x


    def n_grad(self,n):
        x = self.xs[n]
        y = self.ys[n]

        dc_dx = [0]*self.lenght

        all_x = [x]
        for w,b in zip(self.ws,self.bs):
            x = sig(w@x+b)
            all_x.append(x)

        dc_dx[-1] = x-y

        for L in range(self.lenght-2,-1,-1):
            dc_dx[L] = self.ws[L].T@(dc_dx[L+1]*d_sig(all_x[L+1]))
            self.b_grad[L] = dc_dx[L+1]*d_sig(all_x[L+1])
            self.w_grad[L] = self.b_grad[L]@all_x[L].T


    def gradient(self):

        for n in range(len(self.xs)):

            self.n_grad(n)
            w = self.ws-self.w_grad
            b = self.bs-self.b_grad

        return w,b









