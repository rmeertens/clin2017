
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent,Embedding
import numpy as np
from keras.layers import Input,LSTM, Activation, Bidirectional, Flatten, Reshape
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop

import seq2seq
from seq2seq.models import AttentionSeq2Seq


from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

class PercentageDataSplitter:
    def __init__(self, percentage):
        self.split_percentage = percentage

    def split_data(self, data):
        split_point = int(len(data) * self.split_percentage)
        data_train, data_test = data[:split_point], data[split_point:]
        return data_train, data_test
    


next_number_assigned_to_char = 1
character_number_dictionary = dict()
character_number_dictionary["+"] = 0
number_character_dictionary = ['+']
def word_to_numbers(word):
    global next_number_assigned_to_char
    global character_number_dictionary
    numbers = []
    for letter in word:
        if letter not in character_number_dictionary:
            character_number_dictionary[letter] = next_number_assigned_to_char
            number_character_dictionary.append(letter)
            next_number_assigned_to_char +=1
        numbers.append(character_number_dictionary[letter])
        
    return numbers

Xdata = []
Ydata = []
longest_word_input_length = 0
longest_word_output_length = 0
with open("clin2017/lexicon.txt") as lexicon_file:
    for two_words in lexicon_file:
        old_spelling, new_spelling = two_words[:-1].split()
        old_spelling_ids = word_to_numbers(old_spelling)
        new_spelling_ids = word_to_numbers(new_spelling)
        if len(old_spelling_ids) > longest_word_input_length:
            longest_word_input_length = len(old_spelling_ids)
        if len(new_spelling_ids) > longest_word_output_length:
            longest_word_output_length = len(new_spelling_ids)
        Xdata.append(old_spelling_ids)
        Ydata.append(new_spelling_ids)
print(longest_word_input_length)
print(longest_word_output_length)
maxlencombined = max(longest_word_input_length,longest_word_output_length)

Xdata = sequence.pad_sequences(Xdata, maxlen=maxlencombined)
Ydata = sequence.pad_sequences(Ydata, maxlen=maxlencombined)



import random




def generate_data():
    new_Xdata = []
    for d in range(100000):
        new_x = [random.randint(0,next_number_assigned_to_char-1) for _ in range(maxlencombined)]
        new_Xdata.append(new_x)
    Xdata = new_Xdata
    Ydata = []
    for d in Xdata:
        Ydata.append(d[::-1])

    print(Xdata[0])
    print(Ydata[0])
    print(Xdata[1])
    print(Ydata[2])
    # create a one-hot vector
    y = np.zeros((len(Ydata), maxlencombined, next_number_assigned_to_char), dtype=np.int32)
    for i, sentence in enumerate(Ydata):
        for t, char in enumerate(sentence):
            y[i, t, char] = 1

    Ydata = y
    return Xdata,Ydata
# print(next_number_assigned_to_char)
# print(len(Ydata))
    
# percentage_splitter = PercentageDataSplitter(0.9)
# Xdata, Xtest = percentage_splitter.split_data(Xdata)
# Ydata, Ytest = percentage_splitter.split_data(Ydata)

# print(len(Ydata))
# print(len(Ydata[0]))
# print(Xdata[0])
# print(Ydata[0])

print('Build model...')

in_network = Input(shape=(maxlencombined,), dtype="int32", name='input_layer')
old_dutch_embedding = Embedding(output_dim=64, input_dim=next_number_assigned_to_char, input_length=maxlencombined)(in_network)
new_dutch_lstm = AttentionSeq2Seq(input_dim=64, input_length=maxlencombined, hidden_dim=1024, output_length=maxlencombined, output_dim=64, depth=1)(old_dutch_embedding)
#old_dutch_lstm = Bidirectional(LSTM(1024))(old_dutch_embedding)
#new_dutch_lstm = RepeatVector(maxlencombined)(old_dutch_lstm) 
#new_dutch_lstm = LSTM(128,return_sequences=True)(new_dutch_lstm)
dutch_output = TimeDistributed(Dense(next_number_assigned_to_char))(new_dutch_lstm)
dutch_output = Activation('softmax')(dutch_output)
model = Model(input=[in_network], output=[dutch_output])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



# model = 
# model.compile(loss='mse', optimizer='rmsprop')

# old_dutch_autoencoder = Flatten()(old_dutch_lstm)

# #print(old_dutch_autoencoder)
# #old_dutch_autoencoder = Dense(2096)(old_dutch_autoencoder)
# #old_dutch_autoencoder = Dense(20*512)(old_dutch_autoencoder)
# #old_dutch_autoencoder = Reshape((20, 512))(old_dutch_autoencoder)

# #new_dutch_lstm = LSTM(1024, return_sequences=True)(old_dutch_lstm)
# new_dutch_lstm = AttentionSeq2Seq(input_dim=64, input_length=maxlen,
#                                   hidden_dim=512, output_length=20, output_dim=512, depth=1)(old_dutch_embedding)

# timed_output = TimeDistributed(Dense(INFOR_VOCAB_SIZE))(new_dutch_lstm)
# timed_new_dutch = TimeDistributed(Dense(INFOR_VOCAB_SIZE))(new_dutch_lstm)

# activation_old_dutch = Activation('softmax')(timed_output)
# activation_new_dutch = Activation('softmax')(timed_new_dutch)


# input_layer = Input()
# model.add(Embedding(output_dim=64, input_dim=next_number_assigned_to_char, input_length=maxlencombined))
# model.add(LSTM(128, input_shape=(maxlencombined,1),return_sequences=True))
# model.add(LSTM(128,input_shape=))
# model.add(TimeDistributed(Dense(next_number_assigned_to_char)))
# model.add(Activation('softmax'))

# print(model.summary())
# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
for i in range(1000):
    Xdata,Ydata = generate_data()
    model.fit(Xdata, Ydata, batch_size=128, nb_epoch=1,verbose=1)

Yanswerpredicted = model.predict(Xtest, verbose=1)


def get_sentence_argmax(l,dict):
    st = ""
    for w in l:
        if np.argmax(w) < len(dict):
            st += dict[np.argmax(w)] + " "
    return st


def get_sentence(l, dict):
    st = ""
    for w in l:
        if w < len(dict):
            st += dict[w] + " "
    return st



for index, line in enumerate(Yanswerpredicted[:50]):
    
    print(get_sentence(Xtest[index],number_character_dictionary))
    print(get_sentence_argmax(line,number_character_dictionary))
    


