import os, sys
import numpy as np
import re
import collections
from six.moves import cPickle
import pdb
from collections import Counter
from matplotlib import pyplot as plt

#folder to save directorirs
SAVE_DIR = './rnn_results'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


'''
Note that all the relevant code for RNN will be in this file itself. 
Hence, this single file will be long. But I'll try to create separate
sections for different sub-parts of this code
'''

#------------------------------------------------------------------------------
'''
Pre-process the data offline and save it in a file
'''
def build_vocab(x):
    print('Total Data (After converting to the list)', len(x))

    x_filtered = []
    for item in x:
        if item in ['',' ']:
            continue
        #fine_items = item.split('.')
        x_filtered.append(item)
    print('Total Data (After filtering the list)', len(x_filtered))

    word_counts = collections.Counter(x_filtered)
     # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    
    # creates mapping of unique characters to integers
    chars = sorted(list(set(vocabulary.keys())))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # Prints the total characters and character vocab size
    n_chars = len(char_to_int)
    n_vocab = len(chars)
    print("The number of total characters are", n_chars)
    print("\nThe character vocab size is", n_vocab)
    #pdb.set_trace()
    return vocabulary, vocabulary_inv

def process_raw_data(data_dir):
    raw_data = ''
    for file_ in os.listdir(data_dir):
        print('Loading file ', file_)
        file_data = open(os.path.join(data_dir, file_),'r').read()
        print('Amount of data read :', len(file_data))
        raw_data = raw_data + ' ' + file_data
    print('Total Data : ', len(raw_data))
    raw_data = re.sub(r'[^\x00-\x7F]+','', raw_data) #Replace non-asii data
    print('Total Data (After Filtering) : ', len(raw_data))
    characters = list(set(raw_data))
    character_to_index = {character:index for index,character in enumerate(sorted(characters))}
    index_to_character = {index:character for index,character in enumerate(sorted(characters))}
    #pdb.set_trace()
    #x = raw_data.split()
    #vocab, words = build_vocab(x)
    print('Vocabulaty size :', len(character_to_index) )

    letter_counts = Counter(raw_data)
    hist_array_X, hist_array_Y = [], []
    for k,v in letter_counts.items():
        hist_array_X.append(k)
        hist_array_Y.append(v)

    
    plt.plot(hist_array_X, hist_array_Y)
    plt.xlabel('character')
    plt.ylabel('frequency')
    plt.title('Histogram of Frequncy of Characters in data-set corpus')
    plt.savefig(os.path.join(SAVE_DIR,'chars_freq.png'))
    plt.close()

    # print("Unique characters in the dictionary: {}".format(len(unique_chars)))
    # print("List of unique characters: ", unique_chars)  

    len_window = 15
    num_points = len(raw_data) // len_window
    
    text_dataset_X, text_dataset_Y  = [], []
    
    for t in range(0, num_points):
        x_val = raw_data[t*len_window: (t+1)*len_window] 
        y_val = x_val[1:] + ' '
        #pdb.set_trace()
       
        x_val_num = []
        for s in x_val:
            x_val_num.append(character_to_index[s])

        y_val_num = []
        for s in y_val:
            y_val_num.append(character_to_index[s])

        # print("x val num: ", x_val_num)
        # print("y val num: ", y_val_num)
        
        text_dataset_X.append(x_val_num)
        text_dataset_Y.append(y_val_num)
    return text_dataset_X, text_dataset_Y, character_to_index, index_to_character
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Logic to make an multiply gate where W x will be performed
class mult_gate:
    def forward(self, W, x):
        return np.dot(W, x)

    def backward(self, W, x, dz):
        # W: d x c
        # x: d x 1
        # dz = c x 1

        # y(dz) = Wx

        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)

        return dW, dx

# Logic to make and and gate where x1 + x2 will be performed 
class add_gate:
    def forward(self, x1, x2):
        return x1 + x2

    def backward(self, x1, x2, dz):
        # x1: d x 1 
        # x2: d x 1
        # dz = d x 1

        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)

        return dx1, dx2

# Logic to compute the sigmoid activation given inputs 
class Sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def backward(self, x, top_diff):
        output = self.forward(x)
        dsig = output * (1.0 - output) * top_diff
        return dsig

# The logic to compute tanh activation
class tanh:
    def forward(self, x):
        return np.tanh(x)
    def backward(self, x, top_diff):
        output = self.forward(x)
        dtan = (1.0 - np.square(output)) * top_diff
        return dtan

# Softmax activation function implementation along with the loss function and its derivative 
class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)
    def loss(self, x, y):
        probs = self.predict(x)
        loss = - np.log(probs[y])
        return loss 
    def diff(self, x, y):
        probs = self.predict(x)
        probs[y] -= 1
        return probs 


# Making the RNN model by combining all the layers 
mulGate = mult_gate()
addGate = add_gate()
activation = tanh()

# This is the wrapper class for the RNN model which will have all the sequential layers required for forward pass and processing 
class RNN_layer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)
        
    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsV = mulGate.backward(V, self.s, dmulv)
        ds = dsV + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)  

        return (dprev_s, dU, dW, dV)


# Initializing the single layer model, basically this model will store nothing but the U, V and W matrices and will update them when required 
class RNN:
    def __init__(self, word_dim, hid_dim=128, bptt_truncate=4):
        self.word_dim = word_dim
        self.hid_dim = hid_dim
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(-np.sqrt(1./self.word_dim), np.sqrt(1./self.word_dim), (self.hid_dim, self.word_dim))
        self.V = np.random.uniform(-np.sqrt(1./self.hid_dim), np.sqrt(1./self.hid_dim), (self.word_dim, self.hid_dim))
        self.W = np.random.uniform(-np.sqrt(1./self.hid_dim), np.sqrt(1./self.hid_dim), (hid_dim, hid_dim))

    # This is the forward function that will take the input from given sequence of words and return the set of layers for the whole model. 
    def forward_propagation(self, x):
        T = len(x)
        layers = []

        # State for the previous time stamp 
        prev_s = np.zeros(self.hid_dim)

        for t in range(0, T):
            layer = RNN_layer()

            # Creating one hot for the word at index t
            input = np.zeros(self.word_dim)
            input[x[t]] = 1

            layer.forward(input, prev_s, self.U, self.W, self.V)
            prev_s = layer.s
            layers.append(layer)

        return layers 

    # This function will perform a forward pass to sample a character sequence given a starting point 
    def infer_propagation(self, T):
        start = np.random.randint(0,37)
        prev_s = np.zeros(self.hid_dim)
        output = Softmax()

        char_predict = []
        for t in range(0, T):
            layer = RNN_layer()

            # Creating one hot for the character input 
            input = np.zeros(self.word_dim)
            input[start] = 1

            layer.forward(input, prev_s, self.U, self.W, self.V)
            # Taking argmax to get the most likely predictions 
            token = np.argmax(output.predict(layer.mulv))
            char_predict.append(token) 

            # Substituting the values for the input character and the token used 
            prev_s = layer.s
            start = token 

        return char_predict 

    # This function will perform a forward pass on all the time stamp of the model and returns a sequence 
    def predict(self, x):
        output = Softmax()
        layers = self.forward_propagation(x)
        output = [np.argmax(output.predict(layer.mulv)) for layer in layers]
        return output 

        
    # Computing the loss average over all the time durations 
    def calculate_loss(self, x, y):
        assert (len(x) == len(y))
        output = Softmax()
        layers = self.forward_propagation(x)
        loss = 0.0 
        for i, layer in enumerate(layers):
            loss += output.loss(layer.mulv, y[i]) 
        return loss / float(len(y)) 

    # Computing the loss for all the datapoints average over the time instances 
    def calculate_total_loss(self, X, Y):
        loss = 0.0 
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y)) 


    # Backpropagation through time where we can perform backpropagation 
    def bptt(self, x, y):
        assert len(x) == len(y) 
        output = Softmax()
        layers = self.forward_propagation(x) 
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape) 

        T = len(layers) 
        prev_s_t = np.zeros(self.hid_dim)
        diff_s = np.zeros(self.hid_dim) 

        for t in range(0, T):
            dmulv = output.diff(layers[t].mulv, y[t])
            input = np.zeros(self.word_dim)
            input[x[t]] = 1

            dprev_s, dU_t, dW_t, dV_t = layers[t].backward(input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv)      
            prev_s_t = layers[t].s
            dmulv = np.zeros(self.word_dim)

            for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
                input = np.zeros(self.word_dim)
                input[x[i]] = 1
                prev_s_i = np.zeros(self.hid_dim) if i == 0 else layers[i-1].s
                dprev_s, dU_i, dW_i, dV_i = layers[i].backward(input, prev_s_i, self.U, self.W, self.V, dprev_s, dmulv)
                # No need to accumulate dV_t separately as it is will not recur in the previous layers 
                dU_t += dU_i
                dW_t += dW_i

            dV += dV_t
            dU += dU_t
            dW += dW_t

        return (dU, dW, dV )
 
    # This function will perform the update step on the model parameteres 
    def sgd_step(self, x, y, learning_rate):
        dU, dW, dV = self.bptt(x, y)
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW

    # This is the main function to train the RNN model 
    def train(self, X_train, Y_train, X_val, Y_val, learning_rate=0.005, nepochs=100, evaluate_loss_after=1):
        num_examples_seen = 0
        losses = []

        # print("x train , y train , x val , y val: ", len(X_train), len(Y_train), len(X_val), len(Y_val))

        for epoch in range(nepochs):
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_total_loss(X_val, Y_val)
                # losses.append((num_examples_seen, loss))
                losses.append(loss)
                print("Loss after examples =%d for epoch epoch=%d: %f" % (num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                # if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                #     learning_rate = learning_rate * 0.5
                #     print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            if epoch % 5 == 0:
                print("losses:", losses)
                plt.plot(losses)
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.title('Loss on test set')
                plt.savefig(os.path.join(SAVE_DIR,'loss_{}.png'.format(epoch)))
                plt.close()
                # Sampling some text from the model 
                for i in range(10):
                    text = generate_text(rnn_model, index_to_char, T=100)
                    print("Sampled text after training: " , text)

            #Code   
            for i in range(0, len(X_train)):
                self.sgd_step(X_train[i], Y_train[i], learning_rate)
                num_examples_seen += 1

        return losses 
###########

# This function will sample text from the rnn model 
def generate_text(rnn, ind_to_char, T):
    sample = rnn.infer_propagation(T)
    sampled_text = ''
    for id in range(0,T):
        char_id = sample[id]
        char = ind_to_char[char_id]

        sampled_text += (char)

    return str(sampled_text) 


if __name__=="__main__":
    #TO DO : Add parse arguments later
    #process data
    X,Y, char_to_index, index_to_char = process_raw_data('/data3/ankit/Coursework/DLCV/Assignment_2/text_data')
    print('Size of data = ', len(X))

    #Split Data
    train_end = 5000 #int(0.7 * len(X))
    X_train = X[0:train_end]#X[2000:3000]
    Y_train = Y[0:train_end]#Y[2000:3000]

    X_test = X[train_end:6000]#len(X)-1]#X[3000:4000]
    Y_test = Y[train_end:6000]#len(X)-1]#Y[3000:4000]

    print('Size of training data :', len(X_train))
    print('Size of test data :', len(X_test))

    # EValuation of the model 
    word_dim = len(char_to_index)
    hidden_dim = 100

    np.random.seed(10)
    rnn_model = RNN(word_dim, hidden_dim)

    # print("Running a single step of sgd")
    # rnn.sgd_step(x_train, y_train, 0.005)#
    # print("Training for single sgd step is complete with lr 5/1000")

    print("Training the all models with sgd ") 
    losses = rnn_model.train(X_train[:], Y_train[:], X_test, Y_test, learning_rate=0.001, nepochs=30, evaluate_loss_after=1)

    print("losses:", losses)
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss on test set')
    plt.savefig(os.path.join(SAVE_DIR,'loss.png'))
    plt.close()


    # Sampling some text from the model 
    for i in range(10):
        text = generate_text(rnn_model, index_to_char, T=100)
        print("Sampled text after training: " , text)