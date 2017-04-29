import yahoo_finance as yahoo
import pandas as pd
import numpy as np

# returive stock data using yahoo Finance API and return a dataFrame
def retrieveStockData(tickerSymbol, startDate, endDate, fileName):
    '''
    Retrieves data from Yahoo! Finance API
    '''
    try:
        print('Retriving data for ticker _' + tickerSymbol + ' ...')
        historical = yahoo.Share(tickerSymbol).get_historical(startDate, endDate)

        data = pd.DataFrame(historical)
        data = data.drop(['Open', 'Close', 'High', 'Low', 'Volume', 'Symbol', 'Date'], axis=1)

        # save as CSV to stop blowing up their API
        data['Adj_Close'].to_csv(fileName, index=False)

        print('Wholesale customers dataset has {} samples with {} features each.'.format(*data.shape))
        return data
    except:
        print('Dataset could not be loaded. Is the dataset missing?')


#retrieveStockData('IBM', '1962-01-02', '2017-04-27', './data/lstm/IBM.csv')
retrieveStockData('AMZN', '1997-05-15', '2016-04-27', './data/lstm/AMZN.csv')