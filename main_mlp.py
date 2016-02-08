import gzip
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random



#---------- Preactivation class definition (before transformation) ----------#

class Preactivation(object):

    def __init__(self, in_dim, out_dim):
        # Weight initialization
        a = 1/np.sqrt(out_dim)
        self.W = np.random.uniform(low=-a, high=a, size=[in_dim, out_dim])
        self.b = np.random.uniform(-0.01, 0.01, out_dim)
        self.W_grad = []
        self.b_grad = []
        self.gradient_acti_func_ = []
        print " preactivation"

    def parameter_gradient(self, input_, gradient_acti_func):
        # Will use parameter_gradient in update_parameters (check if this takes too much memory)

        W_grad_ = input_.T.dot(gradient_acti_func)
        b_grad_ = gradient_acti_func.sum(axis=0)

        self.gradient_acti_func_.append(gradient_acti_func)
        self.W_grad.append(W_grad_)             # store my W_grad for update
        self.b_grad.append(b_grad_)             # store my b_grad for update
        """Debug
        print self
        print "size of W", np.shape(self.W_grad)
        """
        l = len(self.gradient_acti_func_)
        new_gradient_acti_func = self.gradient_acti_func_[l-1]
        old_gradient_acti_func = self.gradient_acti_func_[l-2]
        if new_gradient_acti_func.all() == old_gradient_acti_func.all():
            print "No update on gradient_acti_func"

    def update_parameters(self, learning_rate, momemtum_rate):
        length_W = len(self.W_grad)
        length_b = len(self.b_grad)
        #print "length_W", length_W
        #print "lenght_b", lenght_b
        new_W_grad = self.W_grad[length_W-1]
        old_W_grad = self.W_grad[length_W-2]
        new_b_grad = self.b_grad[length_b-1]
        old_b_grad = self.b_grad[length_b-2]


        if new_W_grad.all() == old_W_grad.all():
            print "No update on W_grad"
        # update weights after backpropagation gradient computation and using momentum
        self.W = self.W - learning_rate * new_W_grad - momemtum_rate * old_W_grad
        # update bias after backpropagation gradient computation and using momentum
        self.b = self.b - learning_rate * new_b_grad - momemtum_rate * old_b_grad

    def forwardpropagation(self, input_):

        if np.shape(self.W)[0] != np.shape(input_)[1]:
            print "Input[1] dimensions not compatible with W[0]"

        return input_.dot(self.W) + self.b     # y = x.W + b

    def backpropagation(self, gradient_wrt_output):
        """ Debug:
        print "Preactivation"
        print "gradient_wrt_output:", np.shape(gradient_wrt_output)
        print "self.W is ", np.shape(self.W)
        print "output", np.shape(gradient_wrt_output.dot(self.W.T))
        """
        return gradient_wrt_output.dot(self.W.T)


#---------- Activation Functions class definition (hidden layer and output layer) ----------#

# if we want to do linear regression in hidden layer
class Linear(object):

    def __init__(self, in_dim, out_dim):
        print "weight initialisation is called"
        self.preactivation = Preactivation(in_dim, out_dim)

    def forwardpropagation(self, input_):
        preactivated_input = self.preactivation.forwardpropagation(input_)
        self.output = preactivated_input                                  # g(x) = x
        return self.output

    def backpropagation(self, input_, gradient_wrt_output):
        gradient_acti_func = np.ones_like(gradient_wrt_output)            # d(g(x)/dx = 1
        """ Debug:
        print "grad shape", np.shape(gradient_acti_func)
        print "input shape", np.shape(input_)
        print "input gradient_wrt_output", np.shape(gradient_wrt_output)
        """
        self.preactivation.parameter_gradient(input_, gradient_acti_func) # keeping the gradient of the activation funtion
        return self.preactivation.backpropagation(gradient_acti_func) # outputs at preactivation


# if we want logistic regression in hidden layer
class Sigmoid(object):

    def __init__(self, in_dim, out_dim):
        self.preactivation = Preactivation(in_dim, out_dim)

    def forwardpropagation(self, input_):
        preactivated_input = self.preactivation.forwardpropagation(input_)
        self.output = 1 / (1 + np.exp(-preactivated_input)) # sigmoid(x) = 1/(1+e^(-x))
        return self.output

    def backpropagation(self, input_, gradient_wrt_output):
        gradient_acti_func = self.output * (1-self.output)  # d(sigmoid(x))/dx = sigmoid(x) * (1- sigmoid(x))
        """ Debug:
        print "Sigmoid"
        print "input gradient_wrt_output", np.shape(gradient_wrt_output)
        print "grad shape", np.shape(gradient_acti_func)
        print "input shape", np.shape(input_)
        print "input self.output", np.shape(self.output)
        """
        self.preactivation.parameter_gradient(input_, gradient_acti_func)  # keeping the gradient of the activation funtion
        return self.preactivation.backpropagation(gradient_acti_func) # outputs at preactivation

# for output layer (softmax)
class Softmax(object):

    def __init__(self, in_dim, out_dim):
        self.preactivation = Preactivation(in_dim, out_dim)

    def forwardpropagation(self, input_):
        preactivated_input = self.preactivation.forwardpropagation(input_)
        exp_in = np.exp(preactivated_input-preactivated_input.max(axis=1, keepdims=True))
        self.output = exp_in/exp_in.sum(axis=1, keepdims = True) # softmax(x_i) = e^(x_i)/sum(e^x_n) for n= 0 to N

        return self.output

    def backpropagation(self, input_, gradient_wrt_output):
        # with Cost_Function.gradient(), I would use Bart's code below
        """
        grad_temp = self.output* gradient_wrt_output
        sum_grad_temp = grad_temp.sum(axis = 1, keepdims = True)
        gradient_acti_func = grad_temp - self.output*sum_grad_temp
        """
        # gradient_acti_func here is equivalent to (y_hat - y)
        # grad_temp = output_*(-y/output_) = -y
        # sum_grad_temp = ~(-1)
        # so gradient_acti_func = -y -output_*(-1) = output_ - y or (y_hat - y)

        # but decided to use the direct approach, i.e. with Cost_Function.error()
        gradient_acti_func = gradient_wrt_output

        self.preactivation.parameter_gradient(input_, gradient_acti_func)  # keeping the gradient of the activation funtion
        return self.preactivation.backpropagation(gradient_acti_func) # outputs at preactivation

# for output layer (relu) -- TBD
class Relu(object):
    def cost():
        pass

#---------- Cost Function class definition ----------#

class Cost_Function(object):

    def cost(self, output_, y):
        return np.sum(-y*np.log(output_))/output_.shape[0]

    def gradient(self, output_, one_hot_y):
        return -one_hot_y/(output_)     # -y / y_hat

    def error(self, output_, one_hot_y):
        return output_ - one_hot_y      # y_hat - y


#---------- Main MLP network class definition ----------#

class MLP(object):

    def __init__(self, layers):
        self.layers = layers

    def forwardpropagation(self, input_):
        # Inputs are stored and then fed into backpropagation
        self.inputs = []
        for layer in self.layers:
            self.inputs.append(input_)
            output_ = layer.forwardpropagation(input_)
            input_ = output_ # output of this layer is my input to my next layer
        return output_

    def backpropagation(self, input_, gradient_wrt_output):
        # chain rule
        for input_, layer in zip(self.inputs[::-1], self.layers[::-1]):
            #print "In MLP class gradient_wrt_output", np.shape(gradient_wrt_output)
            gradient_wrt_input = layer.backpropagation(input_, gradient_wrt_output)
            gradient_wrt_output = gradient_wrt_input
            #print "In MLP class gradient_wrt_input back", np.shape(gradient_wrt_input)
        return gradient_wrt_output

    def update_parameters(self, learning_rate, momemtum_rate):
        for layer in self.layers:
            layer.preactivation.update_parameters(learning_rate, momemtum_rate)


#---------- Creating a matrix of 1s and 0s to represent labels 1 to 10 ----------#

def one_hot(output_, y):
    one_hot_y = np.zeros_like(output_)
    one_hot_y[range(np.shape(y)[0]),y]= 1
    return one_hot_y


#---------- Loading and visualizing image files ----------#

def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        return pickle.load(f)

train_set, valid_set, test_set = load_data()
train_x, train_y = train_set
valid_x, valid_y = valid_set

"""
# Visualizing what the inputs look like
gs = gridspec.GridSpec(4, 4)
ax = []
img = []

for i in xrange(0, 16):
    im = random.uniform(0, 100)
    ax.append(plt.subplot(gs[i]))
    img.append(ax[i].imshow(train_x[im].reshape(28, 28), cmap = cm.Greys_r))
    img[i].axes.get_xaxis().set_visible(False)
    img[i].axes.get_yaxis().set_visible(False)

plt.subplots_adjust(wspace = None, hspace = None)
plt.show()
plt.close('all')
"""

#---------- Define key constants for parameters ----------#

learning_rate = 0.01
momemtum_rate = 0.01
total_epochs = 1
batch_size = 1000
num_batches = int(train_set[0].shape[0] / batch_size)


#---------- Main loop going through epoch and batches----------#

# Available activation Functions
    # Linear(in_dim, out_dim)
    # Sigmoid(in_dim, out_dim)
    # Softmax(in_dim, out_dim)

in_dim = valid_x.shape[1]
network = MLP([Sigmoid(in_dim, 100), Softmax(100, 10)])


for t in xrange(1, total_epochs + 1):

    out_y = network.forwardpropagation(valid_x)
    cost_func = Cost_Function()
    one_hot_y = one_hot(out_y,valid_y)
    cost = cost_func.cost(out_y, one_hot_y)
    print cost

    # Training network
    for n in xrange(num_batches):
        # making minibatches of data
        first_batch = batch_size * n
        last_batch = batch_size * (n + 1)
        batch_train_x = train_x[first_batch:last_batch]
        batch_train_y = train_y[first_batch:last_batch]
        # Feedforward pass
        out_y_train = network.forwardpropagation(batch_train_x)
        one_hot_y = one_hot(out_y_train,batch_train_y)
        gradient = cost_func.error(out_y_train, one_hot_y)
        # Backward pass
        network.backpropagation(batch_train_x, gradient)
        network.update_parameters(learning_rate, momemtum_rate)
