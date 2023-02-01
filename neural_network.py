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
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size) # weights between hidden layer 1 and hidden layer 2
        self.W3 = np.random.randn(self.hidden_size, self.output_size) # weights between hidden layer 2 and output


    def forward_propogation(self, X):
        self.z = np.dot(X, self.W1) # hidden layer 1 pre activation. (1x784) dot (784x16) = (1x16)
        self.a = self.sigmoid(self.z) # hidden layer 1
        self.z2 = np.dot(self.z, self.W2) # hidden layer 2 pre activation (1x16) dot (16x16) = (1x16)
        self.a2 = self.sigmoid(self.z2) # hidden layer 2 
        self.z3 = np.dot(self.z2, self.W3) # output pre activation (1x16) dot (16x10) = (1x10)
        self.output = self.sigmoid(self.z3) # output

        return self.output


    def back_propogation(self, X, y, output):
        
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.sigmoid(output, derivative=True)

        self.a2_error = self.output_delta.dot(self.W3.T) # contribution of hidden layer 2 to output eror
        self.a2_delta = self.a2_error * self.sigmoid(self.a2, derivative=True)

        self.a_error = self.a2_delta.dot(self.W2.T) # contribution of hidden layer 1 to hl2 error
        self.a_delta = self.a_error * self.sigmoid(self.a, derivative=True)

        W1_delta = X.T.dot(self.a_delta)
        W2_delta = self.a.T.dot(self.a2_delta)
        W3_delta = self.a2.T.dot(self.output_delta)

        return W1_delta, W2_delta, W3_delta


    def train(self, X, y):
        output = self.forward_propogation(X)
        W1, W2, W3 = self.back_propogation(X, y, output)
        return W1, W2, W3


    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1-x)
        return 1.0 / (1 + np.exp(-x))

        
    def training_batch(self, batch):
        W1_delta = None
        W2_delta = None
        W3_delta = None

        for i in range(len(batch.index)):
            entry = batch.iloc[i]
            W1, W2, W3 = self.train(entry[:784], self.parse_output(int(entry['correct_output'])))
            if W1_delta == None:
                W1_delta = W1
                W2_delta = W2
                W3_delta = W3
            else:
                W1_delta += W1
                W2_delta += W2
                W3_delta += W3
        
        self.W1 += W1_delta
        self.W2 += W2_delta
        self.W3 += W3_delta


    def parse_output(self, num):
        out = np.zeros(10)
        out[num] = 1
        return out

    def stochastic_gradient_descent(self, training_df, batch_size):
        
        # the data is already randomly shuffled, so we chilling on that front
        
        if (len(training_df) % batch_size != 0):
            raise Exception("Need evenly split batches")
        
        num_batches = int(len(training_df) / batch_size)

        for i in range(num_batches):
            print(f"Starting batch number: {i}")
            batch = training_df.iloc[batch_size * i : batch_size * (i+1)].reset_index()
            self.training_batch(batch)


def main():

    data = pd.read_pickle("./mnist.pkl")
    training_data = data[0]
    training_data_df = pd.DataFrame(data=training_data[0])
    training_data_df['correct_output'] = training_data[1]
    
    neural_network = Network()
    neural_network.stochastic_gradient_descent(training_data_df, 500)


if __name__ == '__main__':
    main()

