# coding: utf-8

# If you change configuration then you might need to reconvert the dataset (delete ../data directory and run main.py again). 

PUNCTUATIONS = {u" ": 0, u"。": 1, u"，": 2,u"？": 3}
VOCABULARY_FILE = "/Users/mayili/Documents/intern/NLP/punctuation/punctuator-master/raw_data/vocab.txt"

RANDOM_SEED = 1
SHOW_WPS = True # Show training progress in % and speed in words per second (should be turned of when output is logged into a file)
BATCH_SIZE = 128
GATE_ACTIVATION = "Sigmoid"
PHASE1 = {
            "HIDDEN_LAYER":2,
            "MAX_EPOCHS": 20,
            "LEARNING_RATE": 0.1,
            "MIN_IMPROVEMENT": 1.003,
            "PROJECTION_SIZE": 100,
            "HIDDEN_SIZE": 100,
            "HIDDEN_ACTIVATION": "Tanh",
            "BPTT_STEPS": 15,
            "TRAIN_DATA": ["/Users/mayili/Documents/intern/NLP/punctuation/punctuator-master/raw_data/train.txt"],
            "DEV_DATA": ["/Users/mayili/Documents/intern/NLP/punctuation/punctuator-master/raw_data/dev.txt"],
         }

PHASE2 = {
            "MAX_EPOCHS": 20,
            "LEARNING_RATE": 0.1,
            "MIN_IMPROVEMENT": 1.003,
            "HIDDEN_SIZE": 100,
            "HIDDEN_ACTIVATION": "Tanh",
            "BPTT_STEPS": 15,
            "TRAIN_DATA": ["/Users/mayili/Documents/intern/NLP/punctuation/punctuator-master/raw_data1/pauses.train.txt"],
            "DEV_DATA": ["/Users/mayili/Documents/intern/NLP/punctuation/punctuator-master/raw_data1/pauses.dev.txt"],
            "USE_PAUSES": True
         }