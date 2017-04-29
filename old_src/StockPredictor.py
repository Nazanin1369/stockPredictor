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

    def __init__(self):
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
    def prepareData(self, predictDate, metric = 'Adj_Close', sequenceLength=5):

        # number of days to predict ahead
        predictDate = dt.datetime.strptime(predictDate, "%Y-%m-%d")
        endDate = dt.datetime.strptime(self.endDate, "%Y-%m-%d")

        #this pandas gets the number of business days ahead, within reason ( i.e. doesn't know about local market
        #public holidays, etc)
        self.numBdaysAhead = abs(np.busday_count(predictDate, endDate))
        print ("business days ahead", self.numBdaysAhead)

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

        # print ("NEW DATA", new_data.tail(1))
        #scale the data for the Linear Regression, SVR and Neural Net
        self.scaler = preprocessing.StandardScaler().fit(new_data)
        scaled_data = pd.DataFrame(self.scaler.transform(new_data))

        # print ("SCALED DATA", scaled_data.tail(1))
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
        print ("R2 score for training set (Linear Regressor): {:.4f}.".format(r2_score(predicttrain, y_train)))
        print ("R2 score for test set (Linear Regressor): {:.4f}.".format(r2_score(predicttest, y_test)))
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
        print ("best svr params", grid_obj.best_params_)

        predicttrain = grid_obj.predict(X_train)
        predicttest = grid_obj.predict(X_test)

        print ("R2 score for training set (SVR): {:.4f}.".format(r2_score(predicttrain, y_train)))
        print ("R2 score for test set (SVR): {:.4f}.".format(r2_score(predicttest, y_test)))
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
        print (model.summary())

        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='rmsprop')
        model.fit(X_train, y_train, nb_epoch=150, batch_size=25, verbose=0)

        predicttest = model.predict(X_test)
        predicttrain = model.predict(X_train)

        print ("R2 score for training set (NN): {:.4f}.".format(r2_score(predicttrain, y_train)))
        print ("R2 score for test set (NN): {:.4f}.".format(r2_score(predicttest, y_test)))
        self.model = model


    def predictNN(self):
        inputSeq = self.scaler.transform(self.final_row_unscaled)
        print ("inputseq", inputSeq)
        predicted = self.model.predict(inputSeq)[0][0]
        print ("Predicted", predicted)
        return predicted


    def trainRNN(self):
        #need unscaled data for the RNN
        colmn = self.data[self.metric]
        colmn = colmn.values
        print ("last 3 cols", colmn[-1])
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
        print('nb sequences:', len(X))

        #convert lists to np arrays
        X = np.array(X)
        y = np.array(y)

        #convert to 3D and 2D tensors for processing by RNN
        X = np.reshape(X, X.shape + (1,))

        y = np.reshape(y, y.shape + (1,))

        print('X_train shape:', X.shape)
        print('X_test shape:', y.shape)

        print ("X and y", X[-1], y[-1])
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)

        model = Sequential()
        model.add(LSTM(50,
                       batch_input_shape=(self.batch_size, self.maxlen, 1),
                       return_sequences=True))
        model.add(LSTM(50,
                       batch_input_shape=(self.batch_size, self.maxlen, 1),
                       return_sequences=False))
        model.add(Dense(1, init='normal', activation='linear'))

        print (model.summary())
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='rmsprop')

        model.fit(X_train, y_train, nb_epoch=150, batch_size=self.batch_size, verbose=0)

        predicttest = model.predict(X_test)
        predicttrain = model.predict(X_train)

        print ("R2 score for training set (RNN): {:.4f}.".format(r2_score(predicttrain, y_train)))
        print ("R2 score for test set (RNN): {:.4f}.".format(r2_score(predicttest, y_test)))
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


