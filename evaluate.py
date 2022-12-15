#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
from __future__ import print_function
import sys
import logging


def read_file(filename):
    file = open(filename, encoding='utf-8')

    tokens = []
    labels = []
    token = []
    label = []

    for line in file.readlines():
        if line.strip() == '':
            tokens.append(token)
            labels.append(label)
            token = []
            label = []
        else:
            line = line.strip().split(' ')
            
            # check columns
            if len(line) != 2: print('ERROR: {}'.format(line))
            else:
                token.append(line[0])
                label.append(line[1])

    #check
    assert(len(tokens) == len(labels))
    return tokens, labels


def compute_precision(guessed_sentences, correct_sentences, labels):
    assert(len(guessed_sentences) == len(correct_sentences))

    confusion_matrix = {}
    confusion_matrix['total'] = {'correctCount': 0, 'count': 0, 'support': 0}
    for label in labels:
        confusion_matrix[label] = {'correctCount': 0, 'count': 0, 'support': 0}
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        
        
        assert(len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if correct[idx][0] == 'B' and correct[idx][2:] in labels: #A new chunk starts
                    confusion_matrix[correct[idx][2:]]['support'] +=1
                    confusion_matrix['total']['support'] +=1
            if guessed[idx][0] == 'B'and guessed[idx][2:] in labels: #A new chunk starts
                ler_class = guessed[idx][2:]
                confusion_matrix[ler_class]['count'] +=1
                confusion_matrix['total']['count'] +=1
#                 count += 1
                
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True
                    
                    while idx < len(guessed) and guessed[idx][0] == 'I': #Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False
                        if correct[idx][0] == 'B'and correct[idx][2:] in labels: #A new chunk starts
                            confusion_matrix[correct[idx][2:]]['support'] +=1
                            confusion_matrix['total']['support'] +=1
                        
                        idx += 1
                    
                    if idx < len(guessed):
                        if correct[idx][0] == 'I': #The chunk in correct was longer
                            correctlyFound = False
                        
                    
                    if correctlyFound:
                        confusion_matrix[ler_class]['correctCount'] +=1
                        confusion_matrix['total']['correctCount'] +=1
                else:
                    idx += 1
            else:  
                idx += 1
   
    precision_list = []
    support_list = []
    for label in labels:
        precision = 0
        if confusion_matrix[label]['count'] > 0:    
            precision = float(confusion_matrix[label]['correctCount']) / confusion_matrix[label]['count']
        precision_list.append(precision)
        support_list.append(confusion_matrix[label]['support'])
    precision = float(confusion_matrix['total']['correctCount']) / confusion_matrix['total']['count']
    precision_list.append(precision)
    support_list.append(confusion_matrix['total']['support'])
    return precision_list, support_list

def classification_report_strict(pathToGoldLabels, pathToPredictions,  labels = ['PER', 'RR', 'AN', 'LD', 'ST', 'STR', 'LDS', 'ORG', 'UN', 'INN', 'GRT', 'MRK', 'GS', 'VO', 'EUN', 'VS', 'VT', 'RS', 'LIT'], output_format='terminal'):

    X, gold_labels = read_file(pathToGoldLabels)
    X_pred, predictions = read_file(pathToPredictions)

    if 'total' not in labels: labels.append('total')
    prec, support = compute_precision(predictions, gold_labels, labels)
    rec, nonesupport = compute_precision(gold_labels, predictions, labels)

    if output_format=='terminal':
        print('{:11}{:12}{:12}{:12}{:18}'.format('Label', 'Precision', 'Recall', 'F1', 'Support'))
        for idx in range(len(labels)):
            f1 = 2*prec[idx]*rec[idx]/(prec[idx]+rec[idx])
            print('{:7}{:10.2f} %{:10.2f} %{:10.2f} %{:11}'.format(labels[idx], prec[idx]*100, rec[idx]*100, f1*100, support[idx]))
        print('\n\n')

    # csv-print with german notation - sep = '|'
    if output_format=='csv':
        print('{}|{}|{}|{}|{}'.format('Label', 'Precision', 'Recall', 'F1', '#'))
        for idx in range(len(labels)):
            f1 = 2*prec[idx]*rec[idx]/(prec[idx]+rec[idx])
            prec_str='{:.3f}'.format(prec[idx]*100).replace(".", ",")
            rec_str='{:.3f}'.format(rec[idx]*100).replace(".", ",")
            f1_str='{:.3f}'.format(f1*100).replace(".", ",")
            print('{}|{} %|{} %|{} %|{}'.format(labels[idx], prec_str, rec_str, f1_str, separator(str(support[idx]))))

def separator(number):
    shift = 0
    for i in range(3,len(number)):
        if i%3 == 0:
            number = number[:-i-shift] + '.' + number[-i-shift:]
            shift += 1
    return number
        
#labels = ['PER', 'RR', 'AN', 'LD', 'ST', 'STR', 'LDS', 'ORG', 'UN', 'INN', 'GRT', 'MRK', 'GS', 'VO', 'EUN', 'VS', 'VT', 'RS', 'LIT']
#classification_report_strict(y, y_pred, labels)

# labels = ['PER', 'RR', 'AN']
# classification_report_strict(y_test, y_test_pred, labels)

# labels = ['LD', 'ST', 'STR', 'LDS']
# classification_report_strict(y_test, y_test_pred, labels)

# labels = ['ORG', 'UN', 'INN', 'GRT', 'MRK']
# classification_report_strict(y_test, y_test_pred, labels)

# labels = ['GS', 'VO', 'EUN']
# classification_report_strict(y_test, y_test_pred, labels)

# labels = ['RL', 'VT']
# classification_report_strict(y_test, y_test_pred, labels)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate.py pathToGoldLabelsFile pathToPredictFile [labelList]")
        exit()

    pathToGoldLabels = sys.argv[1]
    pathToPredictions = sys.argv[2]
    if len(sys.argv) == 4:
        labelList = sys.argv[3]
        classification_report_strict(pathToGoldLabels, pathToPredictions, labelList)
    else:
        classification_report_strict(pathToGoldLabels, pathToPredictions)
