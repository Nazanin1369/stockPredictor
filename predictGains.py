#!/usr/bin/python
import re
import csv
import os
import urllib.request
import sys
import time
import datetime
import codecs
import numpy as np
import matplotlib.pyplot as plt
import lstm

# Global Variables
predictedPrice={}
seq_len  = 50

def plotGains(strategies, numDays):
    gains = {}
    for strategy in strategies:
        gains[strategy] = calculateGains(strategies[strategy], numDays)
    N = len(gains)
    x = range(N)

    width = 1/1.5

    plt.bar(x, gains.values(),width, color='blue')
    width = .35

    plt.bar(x, gains.values(), width=width)
    plt.savefig('strategyGain.png')


def calculateGains(strategy, numDays):
    '''
    Calculates strategy gains based on predictions
    '''
    gain = 0.0
    for k in predictedPrice:
        gain = gain + float(predictedPrice[k][numDays-1])*float(strategy[k])
    print ('> Strategy Gain:'.format(gain))
    return gain

if __name__ == '__main__':
    strategies = {}

    print('> Predicting Google prices...')
    goog_prediction, goog_acc = lstm.calculate_price_movement('GOOG', seq_len)
    print('Google acc ' ,goog_acc )

    print('> Predicting Microsoft prices...')
    msft_prediction, msft_acc = lstm.calculate_price_movement('MSFT', seq_len)
    print('MSFT acc', msft_acc)

    print('> Including accuracy in predictions...')
    goog_prediction = [a * goog_acc for a in goog_prediction]
    msft_prediction = [a * msft_acc for a in msft_prediction]

    predictedPrice['GOOG'] = goog_prediction
    predictedPrice['MSFT'] = msft_prediction

    strategies['harshal']= {'GOOG':0.5,'MSFT' : 0.5}
    strategies['naz']={'GOOG' : 0.2, 'MSFT' :0.8}
    plotGains(strategies, seq_len)



