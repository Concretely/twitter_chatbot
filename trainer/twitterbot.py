import re
import random
import time
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Embedding, RepeatVector, concatenate, \
    TimeDistributed
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import config
import tensorflow.keras
import string
import pickle


print('Library versions:')
print('tensorflow:{}'.format(tf.__version__))
import pandas as pd
print('pandas:{}'.format(pd.__version__))
import sklearn
print('sklearn:{}'.format(sklearn.__version__))
import nltk
print('nltk:{}'.format(nltk.__version__))
import numpy as np
print('numpy:{}'.format(np.__version__))

import config

import string
exclude = set(string.punctuation)


def remove_punctuation(x):
    """
raw_word = re.sub(r'\s([?.!:,"](?:\s|$))', r'\1', raw_word)    Helper function to remove punctuation from a stringraw_word = re.sub(r'\s([?.!:,"](?:\s|$))', r'\1', raw_word)
    x: any string
    """
    try:
        x = ''.join(ch for ch in x if ch not in exclude)
    except:
        pass
    return x


class twitterbot:
    def __init__(self):
        self.analyzer = {}
        self.vocab = {}
        self.reverse_vocab = {}
        self.c = config.params()
        self.best_score = 100

#----------------------I/O stuff consolodated here
    def dump_parameter_file(self, data, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(data, fp)
    
    def load_parameter_file(self, fname):
        with open(fname, 'rb') as fp:
            return pickle.load(fp)
    
    def load_model_from_disk(self, model_file):
        # load weights into new model
        self.model=load_model(model_file)
        print("Loaded model from disk")
        return self.model

    def save(self):
        # serialize weights to HDF5
        self.model.save(self.c.MODEL_FNAME)
        print("Saved model to disk")

    def read_input_data(self, input_file):
        df = pd.read_csv(input_file, sep='\t' ,names=['tweet','response'])
        df.tweet = df.tweet.apply(self.preprocess)
        df.response = df.response.apply(self.preprocess)
        
        self.texts = df.tweet
        self.responses = df.response
        print('t shape: {}'.format(self.texts.shape))
        print('r shape: {}'.format(self.responses.shape))
  
    def load_embeddings_index(self):
        embeddings_index={}
        f = open(self.c.EMBEDDING_FNAME)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index
#------------------------------------------------


    def preprocess(self, raw_word):
        #print(raw_word)
        l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t']
        l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not']
      
        for j, term in enumerate(l1):
            raw_word = str(raw_word).replace(term,l2[j])
        raw_word = raw_word.lower()
        raw_word = raw_word.replace('__unk__', '')
        raw_word = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",raw_word).split())
        return raw_word

    def to_word_idx(self, sentence):
        full_length = [self.vocab.get(tok, self.c.UNK) for tok in self.analyzer(
            sentence)] + [self.c.PAD] * self.c.MAX_MESSAGE_LEN
        return full_length[:self.c.MAX_MESSAGE_LEN]

    def from_word_idx(self, word_idxs):
        return ' '.join(self.reverse_vocab[idx] for idx in word_idxs if idx != self.c.PAD).strip()

    
    


    def build_vocab(self):
        count_vec = CountVectorizer(
            tokenizer=casual_tokenize, max_features=self.c.MAX_VOCAB_SIZE - 3)
        print("Fitting CountVectorizer on X and Y text data...")
        count_vec.fit(self.texts, self.responses)
        self.dump_parameter_file(count_vec, self.c.COUNT_VEC_FNAME)

        self.analyzer = count_vec.build_analyzer()
        self.vocab = {k: v + 3 for k, v in count_vec.vocabulary_.items()}
        self.vocab['__unk__'] = self.c.UNK
        self.vocab['__pad__'] = self.c.PAD
        self.vocab['__start__'] = self.c.START
        # Used to turn seq2seq predictions into human readable strings
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print("Learned vocab of {} items.".format(len(self.vocab)))
        self.dump_parameter_file(self.vocab, self.c.VOCAB_FNAME)

    def build_embedding_matrix(self):
        embeddings_index = self.load_embeddings_index()
        print('Found %s word vectors.' % len(embeddings_index))
        self.embedding_matrix = np.zeros((self.c.MAX_VOCAB_SIZE, self.c.EMBEDDING_SIZE))

        i = 0
        j = 0
        for word in self.vocab:
            embedding_vector = embeddings_index.get(word)
            
            if embedding_vector is not None:
                #print('adding word {} to matrix'.format(word))
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
                j += 1
            i += 1
        print('Embedded {} words'.format(j))



    def build_vectors(self):
        print("Calculating word indexes for X...")
        x = pd.np.vstack(self.texts.apply(self.to_word_idx).values)
        print("Calculating word indexes for Y...")
        y = pd.np.vstack(self.responses.apply(self.to_word_idx).values)

        all_idx = list(range(len(x)))
        train_idx = set(random.sample(all_idx, int(0.8 * len(all_idx))))
        test_idx = {idx for idx in all_idx if idx not in train_idx}

        self.train_x = x[list(train_idx)]
        self.test_x = x[list(test_idx)]
        self.train_y = y[list(train_idx)]
        self.test_y = y[list(test_idx)]

        print('Training data of shape {} and test data of shape {}.'.format(
            self.train_x.shape, self.test_x.shape))

    def build_model(self):
        shared_embedding = Embedding(
            output_dim=self.c.EMBEDDING_SIZE,
            input_dim=self.c.MAX_VOCAB_SIZE,
            input_length=self.c.MAX_MESSAGE_LEN,
            weights=[self.embedding_matrix],
            name='embedding',
        )

        # ENCODER
        encoder_input = Input(
            shape=(self.c.MAX_MESSAGE_LEN,),
            dtype='int32',
            name='encoder_input',
        )
        embedded_input = shared_embedding(encoder_input)

        # No return_sequences - since the encoder here only produces a single value for the
        # input sequence provided.
        encoder_rnn = LSTM(
            self.c.CONTEXT_SIZE,
            name='encoder',
            dropout=self.c.DROPOUT

        )

        context = RepeatVector(self.c.MAX_MESSAGE_LEN)(
            encoder_rnn(embedded_input))

        # DECODER
        last_word_input = Input(
            shape=(self.c.MAX_MESSAGE_LEN, ),
            dtype='int32',
            name='last_word_input',
        )

        embedded_last_word = shared_embedding(last_word_input)
        # Combines the context produced by the encoder and the last word uttered as inputs
        # to the decoder.
        decoder_input = concatenate([embedded_last_word, context], axis=2)

        # return_sequences causes LSTM to produce one output per timestep instead of one at the
        # end of the intput, which is important for sequence producing models.
        decoder_rnn = LSTM(
            self.c.CONTEXT_SIZE,
            name='decoder',
            return_sequences=True, 
            dropout=self.c.DROPOUT

        )

        decoder_output = decoder_rnn(decoder_input)

        # TimeDistributed allows the dense layer to be applied to each decoder output per timestep
        next_word_dense = TimeDistributed(
            Dense(int(self.c.MAX_VOCAB_SIZE / 2), activation='softmax'),
            name='next_word_dense',
        )(decoder_output)

        next_word = TimeDistributed(
            Dense(self.c.MAX_VOCAB_SIZE, activation='softmax'),
            name='next_word_softmax'
        )(next_word_dense)

        self.model = Model(inputs=[encoder_input, last_word_input], outputs=[next_word])
        optimizer = Adam(lr=self.c.LEARNING_RATE, clipvalue=0.5)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    def add_start_token(self, y_array):
        # Adds the start token to vectors.  Used for training data.
        return np.hstack([
            self.c.START * np.ones((len(y_array), 1)),
            y_array[:, :-1],
        ])

    def binarize_labels(self, labels):
        # Helper function that turns integer word indexes into sparse binary matrices for
        #    the expected model output.
        return np.array([np_utils.to_categorical(row, num_classes=self.c.MAX_VOCAB_SIZE)
                         for row in labels])



    def respond_to(self, text):
        # Helper function that takes a text input and provides a text output.
        input_y = self.add_start_token(
            self.c.PAD * np.ones((1, self.c.MAX_MESSAGE_LEN)))
        idxs = np.array(self.to_word_idx(text)).reshape(
            (1, self.c.MAX_MESSAGE_LEN))
        for position in range(self.c.MAX_MESSAGE_LEN - 1):
            prediction = self.model.predict([idxs, input_y]).argmax(axis=2)[0]
            input_y[:, position + 1] = prediction[position]
        return self.preprocess(self.from_word_idx(self.model.predict([idxs, input_y]).argmax(axis=2)[0]))

    def do_train(self, start_idx, end_idx):
        # Batching seems necessary in Kaggle Jupyter Notebook environments, since
        #    `model.fit` seems to freeze on larger batches (somewhere 1k-10k).

        b_train_y = self.binarize_labels(self.train_y[start_idx:end_idx])
        input_train_y = self.add_start_token(self.train_y[start_idx:end_idx])
        self.model.fit(
            [self.train_x[start_idx:end_idx], input_train_y],
            b_train_y,
            epochs=1,
            batch_size=self.c.BATCH_SIZE
        )

        rand_idx = random.sample(list(range(len(self.test_x))), self.c.BATCH_SIZE)
        test_results = self.model.evaluate(
            [self.test_x[rand_idx], self.add_start_token(
                self.test_y[rand_idx])],
            self.binarize_labels(self.test_y[rand_idx])
        )
        print('Test results:', test_results)
        
        input_strings = [
                "@AmazonHelp Having a problem with my Kindle",
                "@AmazonHelp Where is my Order",
            ]
            
        for input_string in input_strings:
            input_string = self.preprocess(input_string)
            output_string = self.respond_to(input_string)
            print('> "{}"\n< "{}"'.format(input_string, output_string))
        return test_results    

        

    

    def train(self):
        # self.read_csv_data(filename)
        #self.read_sqlite_data('test.db')
        self.read_input_data(self.c.INPUT_FNAME)
        
        num_bad = self.c.BREAK_BAD
        curr_bad = 0
        tmp_score = 0
        self.build_vocab()
        self.build_embedding_matrix()
        #print(type(self.vocab))
        
        self.build_vectors()
        self.build_model()
        #self.load_model_from_disk(self.c.MODEL_FNAME)
        #optimizer = Adam(lr=self.c.LEARNING_RATE)
        #self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        print('do_train')    
        
        for x in range(self.c.NUM_ITERATIONS):
            print('epoch {}===================================================================='.format(x))
            for start_idx in range(0, len(self.train_x), self.c.SUB_BATCH_SIZE):
                curr_bad = 0
                print('start_idx {}, best_score={}'.format(start_idx, self.best_score))    
                tmp_score = self.do_train(start_idx, start_idx+self.c.SUB_BATCH_SIZE)
                if tmp_score < self.best_score:
                    self.best_score = tmp_score
                    curr_bad = 0
                    print('saving self.best_score = {}'.format(self.best_score))
                    self.save()
                else:
                    curr_bad += 1
                    if curr_bad >= num_bad:
                        print('exiting because of curr_bad')
                        break
            else:
                continue
            break
        
        self.save()

   

    def create_responder(self):
        count_vec = self.load_parameter_file(self.c.COUNT_VEC_FNAME)
        self.analyzer = count_vec.build_analyzer()
        self.vocab = self.load_parameter_file(self.c.VOCAB_FNAME)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.model = self.load_model_from_disk(self.c.MODEL_FNAME)
        print(type(self.model))
        
        


        
