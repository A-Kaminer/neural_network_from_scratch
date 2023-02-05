import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    '''
    Neural network class for a one layer network
    '''

    def __init__(self):

        # layer sizes
        self.input_size = 1
        self.output_size = 1
        self.hidden_size = 3

        # weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    
    def forward_propogate(self, X):
        '''
        Takes a set of input vectors and returns the output of those vectors
        through the network.
        '''

        self.z1 = vector_reLu(np.dot(X, self.W1)) # hidden layer
        output = np.dot(self.z1, self.W2) # output

        return output


    def back_propogate(self, X, y, output):
        '''
        Weights to train the network
        '''

        # error and derivative of output
        self.output_error = y - output
        self.output_delta = self.output_error * vector_reLu(output, 
                derivative=True)

        # error and derivative of hidden layer
        self.z1_error = self.output_delta.dot(self.W2.T)
        self.z1_delta = self.z1_error

        W1_delta = X.T.dot(self.z1_delta)
        W2_delta = self.z1.T.dot(self.output_delta)

        return W1_delta, W2_delta


    def train(self, X, y):
        '''
        Trains the model on a batch of data. Updates the weights after running
        through everything to get an accurate step.
        '''

        output = self.forward_propogate(X)
        W1_delta, W2_delta = self.back_propogate(X, y, output)

        self.W1 += W1_delta
        self.W2 += W2_delta


    def test(self, X, y):
        '''
        Test to see how good function is at getting the correct answer.
        '''

        output = self.forward_propogate(X)
        print(f"Percent error: {self.calculate_percent_error(output, y)}")

        '''
        ax = plt.subplot(1,1,1)
        actual_output = ax.plot(X, output, label="Neural Network Output")
        expected_output = ax.plot(X, y, label="Function Output")

        handles, labels = ax.get_legend_handles_labels()

        ax.legend(handles, labels)

        plt.show()
        '''


    def run_batch(self, X, y):
        '''
        Run one training batch and then test network.
        '''
        self.train(X, y)
        self.test(X, y)


    def calculate_percent_error(self, output, y):
        '''
        Calculate the average percent error
        '''
        return  np.sum(np.absolute((y - output) / y)) / len(y)


def reLu(x, derivative=False):
    '''
    reLu function
    '''

    if derivative:
        if x > 0:
            return 1
        else:
            return 0
    else:
        if x > 0:
            return x
        else:
            return 0

def vector_reLu(x, derivative=False):
    for i in range(len(x)):
        x[i][0] = reLu(x[i][0], derivative=derivative)
    return x


def function(x):
    '''
    Function for the function that we try to recreate:
    -x^3 + x^2 + 3x - 1
    '''
    return (-1 * x**(3)) + x**(2) + (3*x) - 1


def main():
    testing_X = np.arange(-10, 10, .01)
    testing_X = np.reshape(testing_X, (len(testing_X), 1))
    testing_y = function(testing_X)
    testing_y = np.reshape(testing_y, (len(testing_y), 1))

  #  plt.plot(testing_X, testing_y)
   # plt.show()

    neural_network = NeuralNetwork()

    for i in range(10):
        neural_network.run_batch(testing_X, testing_y)




if __name__ == '__main__':
    main()
