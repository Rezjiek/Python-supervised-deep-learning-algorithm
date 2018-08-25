import numpy as np
import matplotlib.pyplot as plt


#sigmoid squishification function
def sig(x):
  return 1/(1+np.exp(-x))


#derivative of the sigmoid function
def d_sig(x):
  return 1/(2+2*np.cosh(x))


"""
The network and gradient descent in a class.

all weights -> self.ws
all biases -> self.bs

weight gradient -> w_grad
bias gradient -> b_grad

all inputs -> self.xs
all (corresponding) outputs -> self.ys
"""
class Network:

  def __init__(self,shape=()):
    self.shape = shape

    self.ws = np.array([np.ones((i,j)) for i,j in zip(shape[1:],shape)])
    self.bs = np.array([np.ones((i,1)) for i in shape[1:]])

    self.w_grad = np.array([np.ones((i,j)) for i,j in zip(shape[1:],shape)])
    self.b_grad = np.array([np.ones((i,1)) for i in shape[1:]])

    self.xs = self
    self.ys = self



  #function where you pass an input (x) through the neural network
  def neur_sum(self,x):

    for w,b in zip(self.ws,self.bs):

      x = sig(w@x+b)

    return x



  #cost function
  #so it gives back how well the network is doing, higher means it's doing worse and vise versa
  def cost(self,n):

    out = 0
    for x,y in zip(self.neur_sum(self.xs[n]),self.ys[n]):
      out += np.sum((x-y)**2)

    return out



  #creates the necessary numbers used for calculating all partials (for the gradient)
  def create_partials(self,n):

    a = np.array([np.zeros((i)) for i in self.shape])
    dz = np.array([np.zeros((i)) for i in self.shape[1:]])

    t = 0
    a[0] = self.xs[0]
    for w,b in zip(self.ws,self.bs):
      t += 1
      a[t] = sig(w@a[t-1]+b)
      dz[t-1] = d_sig(w@a[t-1]+b)

    return a,dz



  #calculates the gradient for all weights and biases
  def gradient(self,n):
    a,dz = self.create_partials(n)

    dc_dx = np.array([np.zeros((i)) for i in self.shape])
    max_L = len(self.shape)-1


    for i in range(len(dc_dx[max_L])):
      dc_dx[max_L][i] = 2*(a[max_L][i]-self.ys[n][i][0])


    for L in range(max_L,0,-1):
      for t in range(len(dc_dx[L-1])):

        dc_dx[L-1][t] = sum([ dc_dx[L][k]*dz[L-1][k]*self.ws[L-1][k][t] for k in range( len(dc_dx[L]) )])

      for row in range(len(self.b_grad[L-1])):
        self.b_grad[L-1][row][0] = dc_dx[L][row]*dz[L-1][row]

        for col in range(len(self.w_grad[L-1][row])):
          self.w_grad[L-1][row][col] = self.b_grad[L-1][row][0]*a[L-1][col]

    return self.w_grad,self.b_grad



  #keeps lowering the cost by subtracting the gradient (gradient descent)
  def lower_cost(self,loops):
    c = 1

    for i in range(loops):
      for n in range(len(self.xs)):
        w,b = c*self.gradient(n)
        self.ws -= w
        self.bs -= b

    return self.ws,self.bs






















