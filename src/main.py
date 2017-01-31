import yahoo_finance as yahoo
import pandas as pd
##
## Retrieves data with Yahoo finanace API and saves it in an csv format
##
def retrieveStockData(symbol):
    print("Retriving data for ticker _" + symbol + "_ .....")
    target_data = yahoo.Share(symbol).get_historical("2015-01-01", "2017-01-01")
    target_data_df = pd.DataFrame(target_data)
    target_data_df.to_csv("../data/" + symbol + "_stock_data.csv")
    print("Data for ticker _" + symbol + "_ has been saved to ../data/"+ symbol + "_stock_data.csv")

##
## Read retrived stock data into a data frame
##
def getStockDataFrame(symbol):
    return  pd.read_csv("./data/" + symbol + "_stock_data.csv", index_col=None, header=0, parse_dates=["Date"])

##
## Main method
##
def main():
    #read the data for a specific stock
    retrieveStockData("YHOO");
    #Analyze the data
    #build graphs
    #Analyze features
    #Build model
    #...Test


if __name__ == "__main__":
    main()
