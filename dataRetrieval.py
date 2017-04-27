import yahoo_finance as yahoo
import pandas as pd
import numpy as np


def prepareData(data):
    '''
    modifies data for plotting
    '''
    #Date and Symbol columns not required
    data.drop(['Symbol'], axis=1, inplace=True)
    pd.to_datetime(data['Date'])
    df = pd.DataFrame(data)
    df.sort_values(by='Date', ascending=True)
    # make date as an index for pandas data frame for visulizations
    df.set_index('Date', inplace=True)
    return df


# returive stock data using yahoo Finance API and return a dataFrame
def retrieveStockData(tickerSymbol, startDate, endDate, fileName, reloadData=False):
    '''
    Retrieves data from Yahoo! Finance API
    '''
    try:
        if reloadData:
            print('Retriving data for ticker _' + tickerSymbol + ' ...')
            historical = yahoo.Share(tickerSymbol).get_historical(startDate, endDate)

            data = pd.DataFrame(historical)
            data = data.drop(['Open', 'Close', 'High', 'Low', 'Volume', 'Symbol', 'Date'], axis=1)

            # save as CSV to stop blowing up their API
            data['Adj_Close'].to_csv(fileName, index=False)
        else:
            # read the existing csv
            data = pd.read_csv(fileName)
            data = prepareData(data)

        print('Wholesale customers dataset has {} samples with {} features each.'.format(*data.shape))
        return data
    except:
        print('Dataset could not be loaded. Is the dataset missing?')
