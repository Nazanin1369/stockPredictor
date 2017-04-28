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
import pandas as pd
import matplotlib.pyplot as plt
import lstm

# Global Variables
predictedPrice={}
seq_len  = 50
readGains = True

def plotGains(strategies, numDays):
    gains = {}
    if readGains:
        gains = pd.read_csv('gains.csv')
        print(gains.describe())
        gains = gains.set_index('strategyName')
        print(gains.head())

        minGain = gains.min()
        print('> minimum Gain : ', minGain)

        negs = gains.loc[gains.gain < 0].size
        #print('> number of negs', negs)

        if(negs < gains.size and negs > 0.0):
            gains['gain'] = gains['gain'].apply(lambda x: (x - minGain)* 100 )
        elif (negs == gains.size):
            gains['gain'] = gains['gain'].apply(lambda x: (x * (-1))* 100 )
        else:
            print('All positive')

        print(gains.head())

        plt.figure()
        ax = gains.plot.bar()
        plt.axhline(0, color='k')
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel("Gain", fontsize=12)
        plt.savefig('strategyGainDataFrame.png')

        plt.figure()
        gains.plot.hist(orientation='horizontal', cumulative=True, color='k', alpha=0.5, bins=1, rwidth=0.2)
        plt.savefig('strategyhistGainDataFrame.png')

        plt.figure()
        gains.plot.box()
        plt.savefig('strategyBox.png')


    else:

        for strategy in strategies:
            gains[strategy] = [calculateGains(strategies[strategy], numDays)]
            print('>  Gain: ', gains[strategy])

        df = pd.DataFrame.from_dict(gains, orient='columns')
        df.to_csv('gains.csv', index=False)


def calculateGains(strategy, numDays):
    '''
    Calculates strategy gains based on predictions
    '''
    gain = 0.0
    for k in predictedPrice:
        gain = gain + float(predictedPrice[k][numDays-1])*float(strategy[k])
    return gain

if __name__ == '__main__':
    strategies = {}
    strategies['st1']= {'GOOG':0.5,'MSFT' : 0.5}
    strategies['st2']={'GOOG' : 0.2, 'MSFT' :0.8}
    strategies['st3']= {'GOOG':0.9,'MSFT' : 0.1}
    strategies['st4']={'GOOG' : 0.4, 'MSFT' :0.6}

    if(readGains == False):
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

    plotGains(strategies, seq_len)



