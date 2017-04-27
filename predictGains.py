#!/usr/bin/python
import re
import csv
import os
import urllib.request
import sys
import time
import datetime
import codecs
import matplotlib.pyplot as plt
import lstm


import numpy as np
predictedPrice={}
# predictedPrice['GOOG'] = [100,101,102,101]
# predictedPrice['MSFT'] = [50,52,51,51.5]
def plotGains(strategies, numDays,):
    gains = {}
    for strategy in strategies:
        gains[strategy] = calculateGains(strategies[strategy], numDays)
    N = len(gains)
    x = range(N)
    width = 1/1.5
    plt.bar(x, gains.values(),width, color="blue")
    width = .35
    #ind = np.arange(len(gains.values()))
    plt.bar(x, gains.values(), width=width)
    plt.savefig("figure.pdf")
def calculateGains(strategy, numDays):
    gain = 0.0
    for k in predictedPrice:
        print("K: ", k)
        gain = gain + float(predictedPrice[k][numDays-1])*float(strategy[k])
    print (gain)
    return gain

if __name__ == "__main__":
    strategies = {}

    print('> Predicting Google prices...')
    goog_prediction, goog_acc = lstm.calculate_price_movement("GOOG",50)
    print("Google acc " ,goog_acc )

    print('> Predicting Microsoft prices...')
    msft_prediction, msft_acc = lstm.calculate_price_movement("MSFT", 50)
    print("MSFT acc", msft_acc)

    print('> Including accuracy in predictions...')
    goog_prediction = [a * goog_acc for a in goog_prediction]
    msft_prediction = [a * msft_acc for a in msft_prediction]


    predictedPrice["GOOG"] = goog_prediction
    predictedPrice["MSFT"] = msft_prediction


    strategies["harshal"]= {"GOOG":0.5,"MSFT" : 0.5}
    strategies["naz"]={"GOOG" : 0.2, "MSFT" :0.8}
    plotGains(strategies, 50)



