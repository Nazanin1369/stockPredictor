import yahoo_finance as yahoo
import pandas as pd


###### Entry Parameters #######
startDate = '2010-01-01'
endDate = '2017-01-01'
ticker = 'GOOG'

#Used for re-running: stops querying the API if we already have the data
fetchData = False

#Stock Data- first step is to obtain the list of stocks, and then select a stock to run through machine learning
fileName = '../data/stocksData.csv'
###############################

# returive stock data using yahoo Finance API and return a dataFrame
def retrieveStockData():
    if fetchData:
         #get data from yahoo finance fo tickerSymbol
        data = pd.DataFrame(yahoo.Share(ticker).get_historical(startDate, endDate))

        # save as CSV to stop blowing up their API
        data.to_csv(fileName)

        # save then reload as the yahoo finance date doesn't load right in Pandas
        data = pd.read_csv(fileName)
    else:
        # read the existing csv
        data = pd.read_csv(fileName)

    # Fill blank or missing data
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
fileName = "backTest.csv"
if reloadData:
    data = quandl.Dataset(tickerSymbol).data(
        params={'start_date': '2010-01-01', 'end_date': queryDate}).to_pandas()
    # save as CSV to stop blowing up their API
    data.to_csv(fileName)
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
sp = StockPredictor.StockPredictor(apiKey)

#loadData will query the Qandl API with the parameters
sp.loadData(tickerSymbol, startDate, endDate, reloadData=reloadData, fileName='qandlData.csv')
#preparedata does the preprocessing
sp.prepareData(queryDate, metric=metric, sequenceLength=5)

#Linear Regression first
sp.trainLinearRegression()
predicted = sp.predictLinearRegression()
print ("Actual:", actual, "Predicted by Linear Regression", predicted)
print ("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print ("Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted)))


#SVR is used next
sp.trainSVR()
predicted = sp.predictSVR()
print ("Actual:", actual, "Predicted by SVR", predicted)
print ("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print ("Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted)))

#Then Neural Network
sp.trainNN()
predicted = sp.predictNN()
print ("Actual:", actual, "Predicted by NNet", predicted)
print ("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print ("Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted)))


#finally, the RNN
sp.trainRNN()
predicted = sp.predictRNN()
print ("end date price", endDatePrice)
print ("Actual:", actual, "Predicted by RNN", predicted)
print ("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print ("Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted)))

