import numpy as np
import pandas as pd
import warnings

warnings.simplefilter("ignore")

class Network:

    def __init__(self):
        self.input_size = 784
        self.output_size = 10
        self.hidden_size = 16


        # Weights

        self.W1 = np.random.randn(self.input_size, self.hidden_size) # weights between input and hidden layer 1
        self.W2 = np.random.randn(self.hidden_size, self.output_size) # weights between hidden layer and output layer


    def forward_propogation(self, X):
        self.z = np.dot(X, self.W1) # hidden layer  pre activation. (1x784) dot (784x16) = (1x16)
        self.hidden = self.sigmoid(self.z) # hidden layer 
        self.z2 = np.dot(self.hidden, self.W2) # output pre activation (1x16) dot (16x10) = (1x10)
        output = self.sigmoid(self.z2) # output

        return output


    def back_propogation(self, X, y, output):
        
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.sigmoid(output, derivative=True)

        self.hidden_error = self.output_delta.dot(self.W2.T) # contribution of hidden layer to output eror
        self.hidden_delta = self.hidden_error * self.sigmoid(self.hidden, derivative=True)


        W1_delta = X.T.dot(self.hidden_delta)
        W2_delta = self.hidden.T.dot(self.output_delta)

        return W1_delta, W2_delta,


    def train(self, X, y):
        output = self.forward_propogation(X)
        W1, W2 = self.back_propogation(X, y, output)
        return W1, W2


    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1-x)
        return 1.0 / (1 + np.exp(-x))

        

    def parse_output(self, num):
        out = np.zeros(10)
        out[int(num)] = 1
        return out


def main():

    data = pd.read_pickle("./.untracked/mnist.pkl")
    training_data = data[0]
    training_data_df = pd.DataFrame(data=training_data[0])
    training_data_df['correct_output'] = training_data[1]
    
    neural_network = Network()

    observation = training_data_df.iloc[1]
    observation_X = observation[:784]
    observation_y = neural_network.parse_output(observation['correct_output'])

    neural_network.train(observation_X, observation_y)


if __name__ == '__main__':
    main()

