import numpy as np
import pandas as pd
import warnings

class NeuralNetwork:
    '''
    Neural Network class to do all of the neural network things.
    '''

    def __init__(self):
        self.input_size = 784
        self.output_size = 10
        self.hidden_size = 3

        # Weights
        
        # weights between input and hidden layer 1
        self.W1 = np.random.randn(self.input_size, self.hidden_size)

        # weights between hidden layer 1 and hidden layer 2
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size)

        # weights between hidden layer 2 and output layer
        self.W3 = np.random.randn(self.hidden_size, self.output_size)

    def stochastic_gradient_descent(self, training_X, training_y, testing_X, 
            testing_y, batch_size):
        '''
        Stochastic gradient descent. Trains the network and returns the accuracy
        '''

        # number of batches, assuming batch_size divides training data size
        num_batches = int(len(training_y) / batch_size)

        for i in range(num_batches):
            # gets batch
            observation_X = training_X.iloc[100 * i : 100 * (i+1)]
            observation_y = training_y[100 * i : 100 * (i+1)]

            # train using batch
            self.train(observation_X, observation_y)

            # test on testing data
            self.test(testing_X, testing_y, i)


    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1-x)
        else:
            return 1.0 / (1.0 + np.exp(-x))


    def forward_propogate(self, X):
        '''
        Takes a set of vectors and returns the output of those vectors through
        the network.
        '''
        self.z1 = self.sigmoid(np.dot(X, self.W1)) # hidden layer 1
        self.z2 = self.sigmoid(np.dot(self.z1, self.W2)) # hidden layer 2
        output = self.sigmoid(np.dot(self.z2, self.W3)) # output layer

        return output


    def back_propogation(self, X, y, output):

        # error and derivative of output
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid(output, 
                derivative=True)

        # error and derivative from hidden layer 2
        self.z2_error = self.output_delta.dot(self.W3.T)
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, derivative=True)

        # error and derivative from hidden layer 1
        self.z1_error = self.z2_delta.dot(self.W2.T)
        self.z1_delta = self.z2_error * self.sigmoid(self.z1, derivative=True)

        # find the derivative for the weights
        W1_delta = X.T.dot(self.z1_delta)
        W2_delta = self.z1.T.dot(self.z2_delta)
        W3_delta = self.z2.T.dot(self.output_delta)

        return W1_delta, W2_delta, W3_delta

    
    def train(self, X, y):
        '''
        Trains the model on a batch of data. Updates the weights after running
        through everything to get an accurate step.
        '''
        output = self.forward_propogate(X)
        del_W1, del_W2, del_W3 = self.back_propogation(X, y, output)

        self.W1 += del_W1
        self.W2 += del_W2
        self.W3 += del_W3


    def test(self, testing_X, testing_y, iteration):
        '''
        Test to see how accurate the network is.
        '''
        output = self.forward_propogate(testing_X)
        percent_correct, avg_certainty = self.calculate_accuracy(output, 
                testing_y)

        print(f"======= ITERATION {iteration}:\n\
                Percent Accuracy: {percent_correct * 10}%\n\
                Average Certainty: {avg_certainty}% certain\n")

    def calculate_accuracy(self, output, y):
        '''Calculates the accuracy of a test'''
        total_correct = 0
        total_certainty = 0

        for i in range(len(output)):
            correct, certainty = self.determine_correct(output, y)

            if correct:
                total_correct += 1

            total_certainty += certainty

        # calculate percentage correct
        percent_correct = total_correct / len(output)

        # calculate average certainty
        avg_certainty = total_certainty / len(output)

        return percent_correct, avg_certainty


    def determine_correct(self, output, y):
        '''
        Determines whether or not an output is correct and certainty of a 
        single observation.
        Inputs should be vectors
        '''
        choice, certainty = self.choice_and_certainty(output)

        # determines whether or not network is correct
        correct = (choice == y.argmax())
        
        # return correctness and certainty
        return correct, certainty


    def choice_and_certainty(self, output):
        '''
        Decides which number the network chose and how certain
        '''
        return self.choice(output), self.choice_certainty(output)


    def choice(self, output):
        '''
        Determines which number the network chose
        '''
        return output.argmax()


    def choice_certainty(self, output):
        '''
        Determines the certainty of the chosen number
        '''
        return np.amax(output) / output.sum()
        

def parse_num(num):
    '''
    Takes an digit and returns the encoding of that integer in a 10 element
    vector, where the index of the digit is 1 and everything else is 0.
    '''
    out = np.zeros(10)
    out[num] = 1
    return out


def encode_array(arr):
    '''
    Takes an array of digits and returns an array of the encoding of those digits.
    See parse_num(num)
    '''
    new_arr = np.random.randn(len(arr), 10)
    for i in range(len(arr)):
        new_arr[i] = parse_num(arr[i])
    return new_arr


def parse_data(path):
    '''
    Gets the from the pkl file and prepares and returns it.
    '''
    data = pd.read_pickle(path)
    training_data = data[0]
    testing_data = data[1]

    training_X = pd.DataFrame(data=training_data[0])
    training_y = encode_array(training_data[1])

    testing_X = pd.DataFrame(data=testing_data[0]).iloc[30:130]
    testing_y = encode_array(testing_data[1])[30:130]

    print("LOG: DATA IS PARSED\n")

    return training_X, training_y, testing_X, testing_y


def main():
    warnings.simplefilter("ignore")

    BATCH_SIZE = 100
    DATA_PATH = "./.untracked/mnist.pkl"

    training_X, training_y, testing_X, testing_y = parse_data(DATA_PATH)

    neural_network = NeuralNetwork()

    while True:

        neural_network.stochastic_gradient_descent(training_X, training_y, testing_X,
                testing_y, BATCH_SIZE)



if __name__ == '__main__':
    main()
