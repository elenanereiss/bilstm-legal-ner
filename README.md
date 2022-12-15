# How to train BiLSTM-CNN-CRF with [German LER Dataset](https://github.com/elenanereiss/Legal-Entity-Recognition)

The Implementation for Sequence Tagging of [BiLSTM-CNN-CRF](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf) is used for the training. See the GitHub Repo for more information.
> This repository contains a BiLSTM-CRF implementation that used for NLP Sequence Tagging (for example POS-tagging, Chunking, or Named Entity Recognition). The implementation is based on Keras 2.2.0 and can be run with Tensorflow 1.8.0 as backend. It was optimized for Python 3.5 / 3.6. It does not work with Python 2.7.


# Create an environment

The best way to use these models is to create an environment in conda with python 3.6:

```
conda create -n ler python=3.6
```

To activate an environment:

```
conda activate ler
```

# Install requirements

```
pip install -r requirements.txt
```

Install tensorflow from wheel. I used the following version for Windows and python 3.6.

```
wget https://raw.githubusercontent.com/fo40225/tensorflow-windows-wheel/master/1.8.0/py36/CPU/sse2/tensorflow-1.8.0-cp36-cp36m-win_amd64.whl
pip install tensorflow-1.8.0-cp36-cp36m-win_amd64.whl
```

# Download dataset
Download the [German LER Dataset](https://github.com/elenanereiss/Legal-Entity-Recognition) splits (train, dev, test) from GitHub and save it in the `data` folder.

```
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_train.conll -P data
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_dev.conll -P data
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_test.conll  -P data
```
# Train

It is possible to use three models for training:
- BiLSTM-CRF: modelName=`blstm-crf`;
- BiLSTM-CRF with character embeddings from BiLSTM: modelName=`char-blstm-crf`;
- BiLSTM-CNN-CRF with character embeddings from CNN: modelName=`blstm-cnn-crf`.

I want to use `char-blstm-crf` because that model gives the best results.

```
python3 train.py char-blstm-crf data/ler_train.conll data/ler_dev.conll data/ler_test.conll
```

# Evaluation

To evaluate the stored model in `models/char-blstm-crf.h5`, we first need preditions from the test split `ler_test.conll`. The predictions are written in a file `ler_test_pred.conll`. After that we can get classification report on entity basis of gold labels and predictions.

```
python3 predict.py models/char-blstm-crf.h5 data/ler_test.conll data/ler_test_pred.conll
python3 evaluate.py data/ler_test.conll data/ler_test_pred.conll
```