import pandas as pd
import glob

# Reads all csvs data into a pandas data frame
def getDataFrame():
    path =r'./data/'
    allFiles = glob.glob(path + '/*.csv')
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0, parse_dates=['Date'])
        list_.append(df)
    frame = pd.concat(list_)
    return frame;



def getYahooDataFrame():
    return  pd.read_csv('./data/YHOO_stock_data.csv',index_col=None, header=0, parse_dates=['Date'])