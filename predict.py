#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
from __future__ import print_function
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys
import logging


def write_predictions(modelPath, inputPath, outputPath):
    inputColumns = {0: "tokens"}


    # :: Prepare the input ::
    sentences = readCoNLL(inputPath, inputColumns)
    addCharInformation(sentences)
    addCasingInformation(sentences)


    # :: Load the model ::
    lstmModel = BiLSTM.loadModel(modelPath)


    dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

    # :: Tag the input ::
    tags = lstmModel.tagSentences(dataMatrix)


    # :: Output to stdout ::
    #for sentenceIdx in range(len(sentences)):
    #    tokens = sentences[sentenceIdx]['tokens']

    #    for tokenIdx in range(len(tokens)):
    #        tokenTags = []
    #        for modelName in sorted(tags.keys()):
    #            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

    #        print("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
    #    print("")
        
    f = open(outputPath, 'w', encoding="utf-8")
    for sentenceIdx in range(len(sentences)):
        tokens = sentences[sentenceIdx]['tokens']

        for tokenIdx in range(len(tokens)):
            tokenTags = []
            for modelName in sorted(tags.keys()):
                tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

            f.write("%s %s\n" % (tokens[tokenIdx], " ".join(tokenTags)))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 predict.py modelPath inputPathToConllFile outputPathToConllFile")
        exit()

    modelPath = sys.argv[1]
    inputPath = sys.argv[2]
    outputPath = sys.argv[3]

    write_predictions(modelPath, inputPath, outputPath)