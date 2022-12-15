from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
from shutil import copyfile

from keras import backend as K

def run_training(model_name, train, dev, test):

    # :: Change into the working dir of the script ::
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # :: Logging level ::
    loggingLevel = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(loggingLevel)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loggingLevel)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # copy files to txt
    copyfile(train, 'data/train.txt')
    copyfile(dev, 'data/dev.txt')
    copyfile(test, 'data/test.txt')

    ######################################################
    #
    # Data preprocessing
    #
    ######################################################
    datasets = {
        'LER':                                                  #Name of the dataset
            {'columns': {0:'tokens', 1:'LER_BIO'},              #CoNLL format for the input data. Column 0 contains tokens, column 2 contains POS and column 2 contains chunk information using BIO encoding
             'label': 'LER_BIO',                                #Which column we like to predict
             'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
             'commentSymbol': None}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
    }

    # :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
    embeddingsPath = 'reimers_german_embeddings.gz'

    # :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
    pickleFile = perpareDataset(embeddingsPath, datasets)


    ######################################################
    #
    # The training of the network starts here
    #
    ######################################################


    #Load the embeddings and the dataset
    embeddings, mappings, data = loadDatasetPickle(pickleFile)

    # Some network hyperparameters
    if model_name == "blstm-crf":
        params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25)}

    elif model_name == "char-blstm-crf":
        params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'LSTM', 'maxCharLength': 50}

    elif model_name == "blstm-cnn-crf":
        params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN', 'maxCharLength': 50}

    else:
        print("existing model names are (1) crf, (2) blstm-crf, (3) cnn-crf")
        exit()

    model = BiLSTM(params)
    model.setMappings(mappings, embeddings)
    model.setDataset(datasets, data)

    #model.storeResults('ler-results.csv') #Path to store performance scores for dev / test

    # pickle the model
    model.modelSavePath = 'models/' + model_name + '.h5'
    model.fit(epochs=100)

if __name__ == "__main__":
    # check
    if len(sys.argv) != 5:
        print("Usage: python3 train.py modelName trainPath devPath testPath")
        exit()
    model_name = str(sys.argv[1])

    # dataset splits
    train = str(sys.argv[2])
    if not os.path.isfile(train):
        print("Error: train file does not appear to exist")
        exit()
    dev = str(sys.argv[3])
    if not os.path.isfile(dev):
        print("Error: dev file does not appear to exist")
        exit()
    test = str(sys.argv[4])
    if not os.path.isfile(test):
        print("Error: test file does not appear to exist")
        exit()
    run_training(model_name, train, dev, test)