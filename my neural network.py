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
    a[0] = self.xs[n]
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


    for i,j in zip(self.neur_sum(self.xs[n]),self.ys[n]):
      dc_dx[max_L] = 2*(i-j)


    for L in range(max_L-1,-1,-1):

      for row in range(self.shape[L]):

        dc_dx[L][row] = sum([dc_dx[L+1][t]*dz[L][t]*self.ws[L][t][row] for t in range(self.shape[L+1])])

      for row in range(self.shape[L+1]):
        self.b_grad[L][row][0] = dc_dx[L+1][row]*dz[L][row]

        for col in range(self.shape[L]):
          self.w_grad[L][row][col] = self.b_grad[L][row][0]*a[L][col]



    return self.w_grad,self.b_grad



  #keeps lowering the cost by subtracting the gradient (gradient descent)
  def lower_cost(self,loops):
    c = 1

    out_cost = []
    for i in range(loops):
      w,b = 0,0
      sum_cost = 0
      for n in range(len(self.xs)):
        sum_cost += k.cost(n)
        w0,b0 = self.gradient(n)
        w,b = w+w0,b+b0

      out_cost.append(sum_cost)
      self.ws -= c*w
      self.bs -= c*b

    return out_cost


def img_to_array(img):

  out = []
  for i,j in np.ndenumerate(img):
    out.append([j])

  return np.array(out)


def all_data(all_urls):
  out = []
  for i in all_urls:
    out.append(img_to_array(plt.imread(i)[:,:,0]))

  return out


urls = [
"C:/Users/Midas/Documents/programmeren/School/Python-supervised-deep-learning-algorithm-master/empty.png",
"C:/Users/Midas/Documents/programmeren/School/Python-supervised-deep-learning-algorithm-master/filled.png"
]


k = Network(shape=(256,15,15,1))


data = all_data(urls)

k.xs = np.array( data )
k.ys = np.array([ [[0]],[[1]] ])


tot_cost = np.array(k.lower_cost(1000))
time = np.arange(1000)

plt.plot(time,tot_cost)


















