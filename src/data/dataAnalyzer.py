import dataReader as dr
import matplotlib
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates

matplotlib.style.use('ggplot')


stocks = dr.getDataFrame();

yahoo_stocks = dr.getYahooDataFrame().sort_values(by='Date')
yahoo_stocks.set_index('Date',inplace=True)
# sort data by data
stocks = stocks.sort_values(by='Date')
# make date as an index for pandas data frame
stocks.set_index('Date',inplace=True)


print(yahoo_stocks.head())

def normalPlot():
    stocks['Close'].plot(figsize=(16, 12));
    plt.savefig('./plots/plot.png', bbox_inches='tight')


normalPlot();