#Rich added a comment to config.py
#Rich is cooler than Grant


class params:
    def __init__(self):
        # 8192 - large enough for demonstration, larger values make network training slower
        self.MAX_VOCAB_SIZE = 2**13
        # seq2seq generally relies on fixed length message vectors - longer messages provide more info
        # but result in slower training and larger networks
        self.MAX_MESSAGE_LEN = 30  
        # Embedding size for words - gives a trade off between expressivity of words and network size
        self.EMBEDDING_SIZE = 100
        # Embedding size for whole messages, same trade off as word embeddings
        self.CONTEXT_SIZE = 100
        # Larger batch sizes generally reach the average response faster, but small batch sizes are
        # required for the model to learn nuanced responses.  Also, GPU memory limits max batch size.
        self.BATCH_SIZE = 500
        # Helps regularize network and prevent overfitting.
        self.DROPOUT = 0.5
        # High learning rate helps model reach average response faster, but can make it hard to 
        # converge on nuanced responses
        self.LEARNING_RATE=0.5

        self.TRAINING_TIME=60

        # Tokens needed for seq2seq
        self.UNK = 0  # words that aren't found in the vocab
        self.PAD = 1  # after message has finished, this fills all remaining vector positions
        self.START = 2  # provided to the model at position 0 for every response predicted

        # Implementaiton detail for allowing this to be run in Kaggle's notebook hardware
        self.SUB_BATCH_SIZE = 5000

        self.CLIPVALUE=0.5

        self.NUM_ITERATIONS=30

        self.COUNT_VEC_FNAME = 'count_vec.pkl'
        self.VOCAB_FNAME = 'vocab.pkl'

        self.MODEL_FNAME='s2s_model.h5'

        self.INPUT_FNAME='/floyd/input/data/tweets.txt'
        self.EMBEDDING_FNAME ='/floyd/input/data/glove.6B.100d.txt'

        self.BREAK_BAD=20
