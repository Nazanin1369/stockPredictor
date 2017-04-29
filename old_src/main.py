import yahoo_finance as yahoo
import pandas as pd
import numpy as np
import StockPredictor


###### Entry Parameters #######
startDate = '2010-01-01'
endDate = '2013-08-19'
ticker = 'GOOG'
metric = 'Adj_Close'
queryDate = '2013-08-20'

#Used for re-running: stops querying the API if we already have the data
reloadData = False

#Stock Data- first step is to obtain the list of stocks, and then select a stock to run through machine learning
fileName = '../data/stocksData.csv'
###############################

# returive stock data using yahoo Finance API and return a dataFrame
def retrieveStockData():
    try:
        if reloadData:
            #get data from yahoo finance fo tickerSymbol
            historical = yahoo.Share(ticker).get_historical(startDate, endDate)
            data = pd.DataFrame(historical)

            # save as CSV to stop blowing up their API
            data.to_csv(fileName, index=False, parse_dates=['Date'])

            # save then reload as the yahoo finance date doesn't load right in Pandas
            data = pd.read_csv(fileName)
        else:
            # read the existing csv
            data = pd.read_csv(fileName)
    except:
        print ("Error")

    #Date and Symbol columns not required
    data.drop(['Symbol'], axis = 1, inplace = True)
    pd.to_datetime(data['Date'])
    # make date as an index for pandas data frame
    #data.set_index('Date',inplace=True)
    #Forward and backfll blank data, better option for this dataset than dropping nulls
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    return data


data = retrieveStockData()

#Forward and backfll blank data, better option for this dataset than dropping nulls
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

#date column not required
data = data.drop('Date', 1)

#whats of interest here is the percentage change from one day to the next
data = data.pct_change()

data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

#locate number of outliers for each column, outlier being 1.5 IQR up or down from upper or lower quartile
outliers = pd.DataFrame(index=data.index)
outliers = pd.DataFrame(np.where((data > 1.5 * ((data.quantile(0.75) - data.quantile(0.25)) + data.quantile(0.75)))
                                 | (data < 1.5 * (data.quantile(0.25) - (data.quantile(0.75)) - data.quantile(0.25))),
                                 1, 0), columns=data.columns)


#transpose the describe so that columns can be added
res = data.describe().transpose()

res['variance'] = data.var()
res['outliers'] = outliers.sum()
res['mean_x_outliers'] = (1/res['outliers'])*res['mean']

print (res.sort_values(by=['mean_x_outliers'],ascending=[False]))

print ("SELECTED STOCK", res.sort_values(by=['mean_x_outliers'],ascending=[False]).transpose().keys()[0])

#now we have the selected stock, play ball.
tickerSymbol = res.sort_values(by=['mean_x_outliers'], ascending=[False]).transpose().keys()[0]


#backtest - query the data, and then query the API to see how close it was to the correct value
fileName = "../data/backTest.csv"
if reloadData:
    data = pd.DataFrame(yahoo.Share(ticker).get_historical(startDate, endDate))
    # save as CSV to stop blowing up their API
    data.to_csv(fileName, index=False)
    # save then reload as the qandl date doesn't load right in Pandas
    data = pd.read_csv(fileName)
else:
    data = pd.read_csv(fileName)

#fetch the actual price so that we can compare with what was predicted
actual = data[metric][data['Date'] == queryDate].values[0]
print("Actual price at date of query", actual)
#the endDatePrice is the price at the end of the data - used for comparison
endDatePrice = data[metric][data['Date'] == endDate].values[0]

#function to define the 'worthiness' of the trade, i.e. risk vs reward.
# Difference in expected return vs actual return on investment.
def varianceOfReturn(endPrice, actualPrice, predictedPrice):
    t1 = abs(actualPrice- endPrice)
    p1 = abs(predictedPrice-actualPrice)
    return (p1/t1)*100.0

#create new stock predictor.  Argument as placeholder for future expansion
sp = StockPredictor.StockPredictor()

#loadData will query the Qandl API with the parameters
sp.loadData(tickerSymbol, startDate, endDate, reloadData=reloadData, fileName=fileName)
#preparedata does the preprocessing
sp.prepareData(queryDate, metric=metric, sequenceLength=5)

#Linear Regression first
print("****** Linear Regression *******")
sp.trainLinearRegression()
predicted = sp.predictLinearRegression()
print ("Actual:", actual, "Predicted by Linear Regression", predicted)
print ("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print ("Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted)))


#SVR is used next
print("****** SVR *******")
sp.trainSVR()
predicted = sp.predictSVR()
print ("Actual:", actual, "Predicted by SVR", predicted)
print ("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print ("Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted)))

#Then Neural Network
print("****** Neural Network *******")
sp.trainNN()
predicted = sp.predictNN()
print ("Actual:", actual, "Predicted by NNet", predicted)
print ("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print ("Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted)))


#finally, the RNN
print("****** RNN *******")
sp.trainRNN()
predicted = sp.predictRNN()
print ("end date price", endDatePrice)
print ("Actual:", actual, "Predicted by RNN", predicted)
print ("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print ("Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted)))

