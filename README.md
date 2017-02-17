
## Data Retrieval

Yahoo Finance API is used to get stocks data for Google over period of seven years. 


```python
import yahoo_finance as yahoo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames


# Show matplotlib plots inline (nicely formatted in the notebook)
%matplotlib inline

# Control the default size of figures in this Jupyter notebook
%pylab inline

pylab.rcParams['figure.figsize'] = (15, 9)

###### Entry Parameters #######
startDate = '2010-01-01'
endDate = '2017-01-01'
ticker = 'GOOG'

#Used for re-running: stops querying the API if we already have the data
fetchData = False

#Stock Data- first step is to obtain the list of stocks, and then select a stock to run through machine learning
fileName = 'data/GoogleData.csv'
###############################

# returive stock data using yahoo Finance API and return a dataFrame
def retrieveStockData():
    try:
        if fetchData:
            print 'Retriving data for ticker _GOOG...'
            historical = yahoo.Share(ticker).get_historical(startDate, endDate)
            
            data = pd.DataFrame(historical)

            # save as CSV to stop blowing up their API
            data.to_csv(fileName, index=False, parse_dates=['Date'])
        else:
            # read the existing csv 
            data = pd.read_csv(fileName)

        #Date and Symbol columns not required
        data.drop(['Symbol'], axis = 1, inplace = True)
        pd.to_datetime(data['Date'])
        # make date as an index for pandas data frame for visulizations
        data.set_index('Date',inplace=True)
        print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
        return data 
    except:
         print "Dataset could not be loaded. Is the dataset missing?"
        

data = retrieveStockData()

display(data.head())

```

    Populating the interactive namespace from numpy and matplotlib
    Wholesale customers dataset has 1762 samples with 6 features each.


    WARNING: pylab import has clobbered these variables: ['indices']
    `%matplotlib` prevents importing * from pylab and numpy



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adj_Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-12-30</th>
      <td>771.820007</td>
      <td>771.820007</td>
      <td>782.780029</td>
      <td>770.409973</td>
      <td>782.750000</td>
      <td>1760200</td>
    </tr>
    <tr>
      <th>2016-12-29</th>
      <td>782.789978</td>
      <td>782.789978</td>
      <td>785.929993</td>
      <td>778.919983</td>
      <td>783.330017</td>
      <td>742200</td>
    </tr>
    <tr>
      <th>2016-12-28</th>
      <td>785.049988</td>
      <td>785.049988</td>
      <td>794.229980</td>
      <td>783.200012</td>
      <td>793.700012</td>
      <td>1132700</td>
    </tr>
    <tr>
      <th>2016-12-27</th>
      <td>791.549988</td>
      <td>791.549988</td>
      <td>797.859985</td>
      <td>787.656982</td>
      <td>790.679993</td>
      <td>789100</td>
    </tr>
    <tr>
      <th>2016-12-23</th>
      <td>789.909973</td>
      <td>789.909973</td>
      <td>792.739990</td>
      <td>787.280029</td>
      <td>790.900024</td>
      <td>623400</td>
    </tr>
  </tbody>
</table>
</div>


## Data Exploration

Let’s briefly describe data features. 
**Open** is the price of the stock at the beginning of the trading day (it need not be the closing price of the previous trading day), 
**high** is the highest price of the stock on that trading day, 
**low** the lowest price of the stock on that trading day, and 
**close** the price of the stock at closing time. 
**Volume** indicates how many stocks were traded. 
**Adjusted close** is the closing price of the stock that adjusts the price of the stock for corporate actions. While stock prices are considered to be set mostly by traders, stock splits (when the company makes each extant stock worth two and halves the price) and dividends (payout of company profits per share) also affect the price of a stock and should be accounted for.

In this section I will explore the data through visualizations and code to understand how each feature is related to the others. I will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.



```python
# Display a description of the dataset
display(data.describe())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adj_Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1762.000000</td>
      <td>1762.000000</td>
      <td>1762.000000</td>
      <td>1762.000000</td>
      <td>1762.000000</td>
      <td>1.762000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>460.416137</td>
      <td>667.732200</td>
      <td>673.341401</td>
      <td>662.013391</td>
      <td>667.988556</td>
      <td>4.051179e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>173.765480</td>
      <td>159.790218</td>
      <td>160.469786</td>
      <td>159.052972</td>
      <td>159.812252</td>
      <td>2.933261e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>217.817563</td>
      <td>436.070761</td>
      <td>442.280760</td>
      <td>433.630737</td>
      <td>438.310758</td>
      <td>7.900000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299.208801</td>
      <td>547.364993</td>
      <td>553.582514</td>
      <td>542.753834</td>
      <td>548.639605</td>
      <td>1.901175e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>437.910342</td>
      <td>614.161057</td>
      <td>619.111038</td>
      <td>609.356057</td>
      <td>613.846067</td>
      <td>3.642400e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>579.082539</td>
      <td>749.452515</td>
      <td>756.218456</td>
      <td>742.882817</td>
      <td>749.962820</td>
      <td>5.165425e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>813.109985</td>
      <td>1220.172036</td>
      <td>1228.882066</td>
      <td>1218.602083</td>
      <td>1226.802152</td>
      <td>2.976060e+07</td>
    </tr>
  </tbody>
</table>
</div>


The first thing I notice here is that the mean values and median values differ a lot. 
That means the distribution should not be a normal distribution. In such a case, median is considered to be more reiable than mean.


```python
#data.set_index('Date',inplace=True)
plt.figure();
data.plot.hist()

```




    <matplotlib.axes._subplots.AxesSubplot at 0x10dc81490>




    <matplotlib.figure.Figure at 0x116bbfe50>



![png](output_5_2.png)


Above displays a histogram of features distribution. We can see here that feature histogram is skewed right or right-tailed. For this type of distribution, we can say the mode should be somewhere around the left and we can argue that the median should be less than the mean.



```python
data.plot.area(stacked=False)
plt.show()
```


![png](output_7_0.png)



```python
data.plot.scatter(x='Volume', y='Adj_Close')
plt.show()
```


![png](output_8_0.png)



```python
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()
```


![png](output_9_0.png)



```python
data["Adj_Close"].plot(grid = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10d871610>




![png](output_10_1.png)


We can observe the price changes of Google stock in seven years. We see in general the price of Google stocks has been increased. We can observe some drops and spikes in visualization also. Mostly in 2013 and 2015.

### Implementation: Selecting Samples

To get a better understanding of stock data and how this data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add three indices to the indices list which will represent the stocks to track. 
It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.


```python
# drop date index to be able to work with data better
data = data.reset_index(drop = True)

# Select three indices to sample from the dataset
indices = [509, 1200, 60]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)

print "Chosen samples of Google stock dataset:"
display(samples)
```

    Chosen samples of Google stock dataset:



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adj_Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>530.592416</td>
      <td>530.592416</td>
      <td>534.562410</td>
      <td>526.292354</td>
      <td>527.002370</td>
      <td>2197600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>324.341201</td>
      <td>649.331085</td>
      <td>649.491107</td>
      <td>639.541092</td>
      <td>645.001127</td>
      <td>3651900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>776.469971</td>
      <td>776.469971</td>
      <td>782.070007</td>
      <td>775.650024</td>
      <td>779.309998</td>
      <td>1461200</td>
    </tr>
  </tbody>
</table>
</div>


### Calculating percentile change


```python
#whats of interest here is the percentage change from one day to the next
data.drop(['Date'], axis = 1, inplace = True)

data = data.pct_change()

data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
    
display(data.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adj_Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.014213</td>
      <td>0.014213</td>
      <td>0.004024</td>
      <td>0.011046</td>
      <td>0.000741</td>
      <td>-0.578343</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.014213</td>
      <td>0.014213</td>
      <td>0.004024</td>
      <td>0.011046</td>
      <td>0.000741</td>
      <td>-0.578343</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.002887</td>
      <td>0.002887</td>
      <td>0.010561</td>
      <td>0.005495</td>
      <td>0.013238</td>
      <td>0.526139</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.008280</td>
      <td>0.008280</td>
      <td>0.004570</td>
      <td>0.005691</td>
      <td>-0.003805</td>
      <td>-0.303346</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.002072</td>
      <td>-0.002072</td>
      <td>-0.006417</td>
      <td>-0.000479</td>
      <td>0.000278</td>
      <td>-0.209986</td>
    </tr>
  </tbody>
</table>
</div>


## Feature Relevance 


```python
for col in list(data.columns.values):    
    # Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    new_data = data.drop(col, axis=1)

    # Split the data into training and testing sets using the given feature as the target
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(new_data, data[col], test_size=0.25, random_state=42)

    # Create a decision tree regressor and fit it to the training set
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)

    # Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    print("{} R^2 score: {:2f}".format(col ,score))
```

    Adj_Close R^2 score: 0.984329
    Close R^2 score: 0.997682
    High R^2 score: 0.682351
    Low R^2 score: 0.717076
    Open R^2 score: 0.437612
    Volume R^2 score: -0.458973


The coefficient of determination, R^2, is scored between 0 and 1, with 1 being a perfect fit. A negative R^2 implies the model fails to fit the data. 

Based on the R^2 score looks like between Open, Close and Adjacent Close prices; Close has the highest relevancy and it is best to use this to determine prediction.
Volume has the lowest score which makes it not being much useful for our predictions since the model couldn't fit trying to predict them using the others.

### Visualize Feature Distributions

To get a better understanding of the dataset, we can construct a scatter matrix of each of the six stock features present in the data. 
If we found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if we believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data.


```python
# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
```


![png](output_20_0.png)


## Identifying outliers
Outliers are data points that are distinctvely separated from other data points any data point more than 1.5 interquartile ranges (IQRs) below the first quartile or above the third quartile.

Outliers has the biggest effect on the mean but not much effect on the median. Effecting mean can lead to effecting variance and then having largest effect on standard deviation.


```python
#locate number of outliers for each column, outlier being 1.5 IQR up or down from upper or lower quartile
outliers = pd.DataFrame(index=data.index)
outliers = pd.DataFrame(np.where(
        (data > 1.5 * ((data.quantile(0.75) - data.quantile(0.25)) + data.quantile(0.75))) |
        (data < 1.5 * (data.quantile(0.25) - (data.quantile(0.75)) - data.quantile(0.25))),1, 0), 
                        columns=data.columns)

#transpose the describe so that columns can be added
res = data.describe().transpose()

display(res)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adj_Close</th>
      <td>1762.0</td>
      <td>-0.000383</td>
      <td>0.015522</td>
      <td>-0.138321</td>
      <td>-0.008367</td>
      <td>-0.000231</td>
      <td>0.007002</td>
      <td>0.091435</td>
    </tr>
    <tr>
      <th>Close</th>
      <td>1762.0</td>
      <td>0.000193</td>
      <td>0.028980</td>
      <td>-0.138321</td>
      <td>-0.008367</td>
      <td>-0.000231</td>
      <td>0.007002</td>
      <td>1.026943</td>
    </tr>
    <tr>
      <th>High</th>
      <td>1762.0</td>
      <td>0.000166</td>
      <td>0.028803</td>
      <td>-0.139055</td>
      <td>-0.006702</td>
      <td>-0.000186</td>
      <td>0.005672</td>
      <td>1.062617</td>
    </tr>
    <tr>
      <th>Low</th>
      <td>1762.0</td>
      <td>0.000185</td>
      <td>0.029012</td>
      <td>-0.124031</td>
      <td>-0.008160</td>
      <td>-0.000929</td>
      <td>0.006225</td>
      <td>1.046402</td>
    </tr>
    <tr>
      <th>Open</th>
      <td>1762.0</td>
      <td>0.000182</td>
      <td>0.029328</td>
      <td>-0.129245</td>
      <td>-0.009158</td>
      <td>-0.000447</td>
      <td>0.007277</td>
      <td>1.045787</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>1762.0</td>
      <td>0.287888</td>
      <td>9.401630</td>
      <td>-0.971152</td>
      <td>-0.188675</td>
      <td>0.018977</td>
      <td>0.256817</td>
      <td>394.358779</td>
    </tr>
  </tbody>
</table>
</div>



```python
res['variance'] = data.var()
res['outliers'] = outliers.sum()
res['mean_x_outliers'] = (1/res['outliers'])*res['mean']


display(res['variance'])
display(res['outliers'])
display(res['mean_x_outliers'])
display(res)
```


    Adj_Close     0.000241
    Close         0.000840
    High          0.000830
    Low           0.000842
    Open          0.000860
    Volume       88.390650
    Name: variance, dtype: float64



    Adj_Close    388
    Close        389
    High         404
    Low          415
    Open         379
    Volume       184
    Name: outliers, dtype: int64



    Adj_Close   -9.863534e-07
    Close        4.962778e-07
    High         4.113046e-07
    Low          4.445897e-07
    Open         4.799700e-07
    Volume       1.564608e-03
    Name: mean_x_outliers, dtype: float64



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>variance</th>
      <th>outliers</th>
      <th>mean_x_outliers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adj_Close</th>
      <td>1762.0</td>
      <td>-0.000383</td>
      <td>0.015522</td>
      <td>-0.138321</td>
      <td>-0.008367</td>
      <td>-0.000231</td>
      <td>0.007002</td>
      <td>0.091435</td>
      <td>0.000241</td>
      <td>388</td>
      <td>-9.863534e-07</td>
    </tr>
    <tr>
      <th>Close</th>
      <td>1762.0</td>
      <td>0.000193</td>
      <td>0.028980</td>
      <td>-0.138321</td>
      <td>-0.008367</td>
      <td>-0.000231</td>
      <td>0.007002</td>
      <td>1.026943</td>
      <td>0.000840</td>
      <td>389</td>
      <td>4.962778e-07</td>
    </tr>
    <tr>
      <th>High</th>
      <td>1762.0</td>
      <td>0.000166</td>
      <td>0.028803</td>
      <td>-0.139055</td>
      <td>-0.006702</td>
      <td>-0.000186</td>
      <td>0.005672</td>
      <td>1.062617</td>
      <td>0.000830</td>
      <td>404</td>
      <td>4.113046e-07</td>
    </tr>
    <tr>
      <th>Low</th>
      <td>1762.0</td>
      <td>0.000185</td>
      <td>0.029012</td>
      <td>-0.124031</td>
      <td>-0.008160</td>
      <td>-0.000929</td>
      <td>0.006225</td>
      <td>1.046402</td>
      <td>0.000842</td>
      <td>415</td>
      <td>4.445897e-07</td>
    </tr>
    <tr>
      <th>Open</th>
      <td>1762.0</td>
      <td>0.000182</td>
      <td>0.029328</td>
      <td>-0.129245</td>
      <td>-0.009158</td>
      <td>-0.000447</td>
      <td>0.007277</td>
      <td>1.045787</td>
      <td>0.000860</td>
      <td>379</td>
      <td>4.799700e-07</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>1762.0</td>
      <td>0.287888</td>
      <td>9.401630</td>
      <td>-0.971152</td>
      <td>-0.188675</td>
      <td>0.018977</td>
      <td>0.256817</td>
      <td>394.358779</td>
      <td>88.390650</td>
      <td>184</td>
      <td>1.564608e-03</td>
    </tr>
  </tbody>
</table>
</div>


## Define worthiness of a trade
Below is a function which determines variance of a trade means how much a trade worth. Which is risk vs reward or in another words, difference in expected return vs actual return on an investment.


```python
def varianceOfReturn(endPrice, actualPrice, predictedPrice):
    t1 = abs(actualPrice- endPrice)
    p1 = abs(predictedPrice-actualPrice)
    return (p1/t1)*100.0

```

## Build an Stock Predictor


```python
from keras.layers import Dense, LSTM
from keras.models import Sequential

from sklearn import svm, metrics, preprocessing
import datetime as dt
import pandas as pd
import numpy as np
import yahoo_finance as yahoo
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Show matplotlib plots inline (nicely formatted in the notebook)
import matplotlib.pyplot as plt


class StockPredictor:

    def __init__(self, APIKey):
        quandl.ApiConfig.api_key = APIKey
        self.tickerSymbol = ''

    #LoadData call queries the API, caching if necessary
    def loadData(self, tickerSymbol, startDate, endDate, reloadData=False, fileName='stockData.csv'):

        self.tickerSymbol = tickerSymbol
        self.startDate = startDate
        self.endDate = endDate

        if reloadData:
            #get data from yahoo finance fo tickerSymbol
            data = pd.DataFrame(yahoo.Share(tickerSymbol).get_historical(startDate, endDate))

            # save as CSV to stop blowing up their API
            data.to_csv(fileName)

            # save then reload as the yahoo finance date doesn't load right in Pandas
            data = pd.read_csv(fileName)
        else:
            data = pd.read_csv(fileName)

        # Due to differing markets and timezones, public holidays etc (e.g. BHP being an Australian stock,
        # ASX doesn't open on 26th Jan due to National Holiday) there are some gaps in the data.
        # from manual inspection and knowledge of the dataset, its safe to take the previous days' value
        data.fillna(method='ffill', inplace=True)
        self.data = data

    #preparedata call does the preprocessing of the loaded data
    #sequence length is a tuning parameter - this is the length of the sequence that will be trained on.
    # Too long and too short and the algorithms won't be able to find any trend - set as 5 days
    # by default, and this works pretty well
    def prepareData(self, predictDate, metric = 'Adjusted Close', sequenceLength=5):

        # number of days to predict ahead
        predictDate = dt.datetime.strptime(predictDate, "%Y-%m-%d")
        endDate = dt.datetime.strptime(self.endDate, "%Y-%m-%d")

        #this pandas gets the number of business days ahead, within reason ( i.e. doesn't know about local market
        #public holidays, etc)
        self.numBdaysAhead = abs(np.busday_count(predictDate, endDate))
        print "business days ahead", self.numBdaysAhead

        self.sequenceLength = sequenceLength
        self.predictAhead = self.numBdaysAhead
        self.metric = metric

        data = self.data
        # Calculate date delta
        data['Date'] = pd.to_datetime(data['Date'])
        data['date_delta'] = (data['Date'] - data['Date'].min()) / np.timedelta64(1, 'D')

        #create the lagged dataframe
        tslag = pd.DataFrame(index=data.index)

        #use the shift function to get the price x days ahead, and then transpose
        for i in xrange(0, sequenceLength + self.numBdaysAhead):
            tslag["Lag%s" % str(i + 1)] = data[metric].shift(1 - i)

        #shift (-2) then corrects the sequence indexes
        tslag.shift(-2)
        tslag['date_delta'] = data['date_delta']

        # create the dataset.  This will take the first [sequenceLength] columns as the data, and the
        # value at end of sequence + number days ahead as the label
        trainCols = ['date_delta']
        for i in xrange(0, sequenceLength):
            trainCols.append("Lag%s" % str(i + 1))
        labelCol = 'Lag' + str(sequenceLength + self.numBdaysAhead)

        # get the final row for predictions
        rowcalcs = tslag[trainCols]
        rowcalcs = rowcalcs.dropna()

        #need an unscaled version for the RNN
        self.final_row_unscaled = rowcalcs.tail(1)


        #due to the way the lagged set is created, there will be some rows with nulls for where
        #the staggering has not worked to to predicting too far back, or ahead.
        #  We can drop these without losing any information as these sequences will be represented in
        #other rows within the dataset
        tslag.dropna(inplace=True)

        label = tslag[labelCol]
        new_data = tslag[trainCols]

        # print "NEW DATA", new_data.tail(1)
        #scale the data for the Linear Regression, SVR and Neural Net
        self.scaler = preprocessing.StandardScaler().fit(new_data)
        scaled_data = pd.DataFrame(self.scaler.transform(new_data))

        # print "SCALED DATA", scaled_data.tail(1)
        self.scaled_data = scaled_data
        self.label = label

    #Linear Regression trainer
    def trainLinearRegression(self):
        lr = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data, self.label, test_size=0.25,
                                                            random_state=42)

        parameters = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}

        grid = GridSearchCV(lr, parameters, cv=None)
        grid.fit(X_train, y_train)
        predicttrain = grid.predict(X_train)
        predicttest = grid.predict(X_test)
        print "R2 score for training set (Linear Regressor): {:.4f}.".format(r2_score(predicttrain, y_train))
        print "R2 score for test set (Linear Regressor): {:.4f}.".format(r2_score(predicttest, y_test))
        self.model = grid

    #predict Linear Regression
    def predictLinearRegression(self):
        inputSeq = self.scaler.transform(self.final_row_unscaled)
        inputSeq = pd.DataFrame(inputSeq)
        predicted = self.model.predict(inputSeq)[0]
        return predicted


    def trainSVR(self):
        clf = svm.SVR()
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data, self.label, test_size=0.25, random_state=42)

        parameters = {'C': [1, 10], 'epsilon': [0.1, 1e-2, 1e-3]}
        r2_scorer = metrics.make_scorer(metrics.r2_score)

        grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=r2_scorer)
        grid_obj.fit(X_train, y_train)
        print "best svr params", grid_obj.best_params_

        predicttrain = grid_obj.predict(X_train)
        predicttest = grid_obj.predict(X_test)

        print "R2 score for training set (SVR): {:.4f}.".format(r2_score(predicttrain, y_train))
        print "R2 score for test set (SVR): {:.4f}.".format(r2_score(predicttest, y_test))
        self.model = grid_obj

    def predictSVR(self):
        inputSeq = self.scaler.transform(self.final_row_unscaled)
        inputSeq = pd.DataFrame(inputSeq)
        predicted = self.model.predict(inputSeq)[0]
        return predicted


    def trainNN(self):
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data.as_matrix(), self.label.as_matrix(), test_size=0.25, random_state=42)

        # create model
        model = Sequential()
        model.add(Dense(270, input_dim=X_train.shape[1], init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='linear'))

        #this is handy to visualise the model
        print model.summary()

        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='rmsprop')
        model.fit(X_train, y_train, nb_epoch=150, batch_size=25, verbose=0)

        predicttest = model.predict(X_test)
        predicttrain = model.predict(X_train)

        print "R2 score for training set (NN): {:.4f}.".format(r2_score(predicttrain, y_train))
        print "R2 score for test set (NN): {:.4f}.".format(r2_score(predicttest, y_test))
        self.model = model


    def predictNN(self):
        inputSeq = self.scaler.transform(self.final_row_unscaled)
        print "inputseq", inputSeq
        predicted = self.model.predict(inputSeq)[0][0]
        print "Predicted", predicted
        return predicted


    def trainRNN(self):
        #need unscaled data for the RNN
        colmn = self.data[self.metric]
        colmn = colmn.values
        print "last 3 cols", colmn[-1]
        self.maxlen = 7

        #self.step = 1
        self.step = self.numBdaysAhead-1

        #batch size tuning parameter
        self.batch_size = 50
        X = []
        y = []
        # as above, need to create data and labels for the RNN.
        for i in range(0, len(colmn) - self.step-self.maxlen):
            X.append(colmn[i: i + self.maxlen])
            y.append(colmn[i + self.step+self.maxlen])
        #print('nb sequences:', len(X))

        #convert lists to np arrays
        X = np.array(X)
        y = np.array(y)

        #convert to 3D and 2D tensors for processing by RNN
        X = np.reshape(X, X.shape + (1,))
        y = np.reshape(y, y.shape + (1,))

        #print('X_train shape:', X.shape)
        #print('X_test shape:', y.shape)

        #print "X and y", X[-1], y[-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25, random_state=42)

        model = Sequential()
        model.add(LSTM(50,
                       batch_input_shape=(self.batch_size, self.maxlen, 1),
                       return_sequences=True))
        model.add(LSTM(50,
                       batch_input_shape=(self.batch_size, self.maxlen, 1),
                       return_sequences=False))
        model.add(Dense(1, init='normal', activation='linear'))

        print model.summary()
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='rmsprop')

        model.fit(X_train, y_train, nb_epoch=150, batch_size=25, verbose=0)

        predicttest = model.predict(X_test)
        predicttrain = model.predict(X_train)

        print "R2 score for training set (RNN): {:.4f}.".format(r2_score(predicttrain, y_train))
        print "R2 score for test set (RNN): {:.4f}.".format(r2_score(predicttest, y_test))
        self.model = model


    def predictRNN(self):
        #need unscaled data for prediction, fetch the last row, with batch size
        cols = self.data[self.metric].tail(self.batch_size*self.maxlen)
        cols = cols.values
        #translate these into the required format
        X = []
        for i in range(0, len(cols)-self.maxlen+1):
            X.append(cols[i: i + self.maxlen])

        #take last row and convert to np array
        inputSeq = np.array([X[-1]])
        #reshape to 3D tensor for RNN
        inputSeq = np.reshape(inputSeq, inputSeq.shape + (1,))
        predicted = self.model.predict(inputSeq)[0][0]
        return predicted



```

    Using TensorFlow backend.



```python

```
