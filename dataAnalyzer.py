import dataReader as dr
import matplotlib
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates

matplotlib.style.use('ggplot')


stocks = dr.getDataFrame();
# sort data by data
stocks = stocks.sort_values(by='Date')
# make date as an index for pandas data frame
stocks.set_index('Date',inplace=True)

print(stocks.dtypes)

print(stocks.head())

def normalPlot():
    stocks['Close'].plot(figsize=(16, 12));
    plt.savefig('./plots/plot.png', bbox_inches='tight')


normalPlot();