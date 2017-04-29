import yahoo_finance as yahoo
import pandas as pd

# Get data from yahoo finance for ticker GOOG
for symbol in ['YHOO', 'AAPL', 'GOOG', 'FIT', 'MSFT', 'LNKD']:
    goog_data = yahoo.Share(symbol).get_historical('2015-01-01', '2017-01-01')
    goog_df = pd.DataFrame(goog_data)
    goog_df.to_csv('./data/' + symbol + '_stock_data.csv')
