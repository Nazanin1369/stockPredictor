import yahoo_finance as yahoo
import pandas as pd


###### Entry Parameters #######
startDate = '2015-01-01'
endDate = '2017-01-01'
ticker = 'GOOG'

#Used for re-running: stops querying the API if we already have the data
fetchData = False

#Stock Data- first step is to obtain the list of stocks, and then select a stock to run through machine learning
fileName = 'data/stocksData.csv'
###############################

# returive stock data using yahoo Finance API and return a dataFrame
def retrieveStockData():
    if fetchData:
        frames = []
        for symbol in ['YHOO', 'GOOG', 'GPRO', 'MSFT', 'LNKD']:
            print('Retriving data for ticker _' + symbol + '_ .....')
            target_data = yahoo.Share(symbol).get_historical(startDate, endDate)
            df = pd.DataFrame(target_data).sort_values(by='Date')
            df = df[['Symbol', 'Date','Open', 'Close', 'Adj_Close', 'High','Low', 'Volume']]
            frames.append(df)

        data = pd.concat(frames)
        # save as CSV to stop blowing up their API
        data.to_csv(fileName, index_col=None, header=0, parse_dates=['Date'])
    else:
        # read the existing csv
        data = pd.read_csv(fileName)

    # Fill blank or missing data
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    return data


df = retrieveStockData()
df.head()

##
## Main method
##
def main():
    #read the data for a specific stock
    retrieveStockData()
    #Analyze the data
    #build graphs
    #Analyze features
    #Build model
    #...Test


if __name__ == "__main__":
    main()
