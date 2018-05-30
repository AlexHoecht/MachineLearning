#############################################################
#IMPORTS
#############################################################
import datetime
import quandl, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



#############################################################
"""
QUICK REFERENCE:
----------------
Features = descriptive attributes
Labels = what is trying to be found

Currently we are trying to predict, or the label, is the future price of
a stock. Therefore our features can be defined as the CurrentPrice (CLOSE),
High-Low Percent (HL_PCT), and the Percent change volatility (PCT_change).

When trying to predict a client's premium for insurance, we want to forcast
the 'Right Now' datapoints that represents 1% of the entire data stream.
Example) If our data represents 100 days of stock prices, the forcast will
show the price 1 day into the future.

In machine learning, we hurt the system if we pass a NaN datapoint to the
classifier. The popular option to handle a NaN datapoint is to replace the
missing data with -99,999. The machine learning classifiers will recognize
this and treat it as an OUTLIER!

svm.SVR provides a default kernal 'rbf'. Kernals are used to simplify the
datastream and speed up processing.
NOTE: svm.SVR also provides other kernals for use such as 'linear', 'poly',
'sigmoid', 'precomputed', or a callable implementation. (See documentaion)

"""
#############################################################
#PLOT
#############################################################
style.use('ggplot')

#############################################################
#DATASTREAM MANIPULATION
#############################################################

#Collect our big data stream and display it in a frame.
df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#From the big data stream we transform specific streams into
#'useful' data for our machine learning.
    
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

#Defining the new data frame with our 'userful' data
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]



#print(df.head())



#Defining the forecasting column that will initially be filled
#with any NaN data.
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

#Defining our LABEL
df['label'] = df[forecast_col].shift(-forecast_out)

#Drop any still NaN datapoints
df.dropna(inplace=True)

#X = features, Y = label
x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

x = preprocessing.scale(x)

x_recent = x[-forecast_out:]

#############################################################
#DEFINING/TRAINING/TESTING THE CLASSIFIER
#############################################################

#Train and test variables for features and labels
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


#Default kernal for svm.SVR() is rbf
clf = svm.SVR()
clf.fit(x_train, y_train)

#Test classifier
confidence = clf.score(x_test, y_test)
#print(confidence)

#Regression algorithm!!! -1 means only available threads are used
clf = LinearRegression(n_jobs=-1)


#VERBOSE FORCAST AND CONFIDENCE!!!!
'''
#Testing types of svm.SVR kernals
for i in ['linear','poly','rbf','sigmoid']:
    clf= svm.SVR(kernel=i)
    clf.fit(x_train, y_train)
    confidence = clf.score(x_test, y_test)
    forecast_set = clf.predict(x_recent)
    print(i,confidence)
    print(forecast_set)
    print()
'''

clf.fit(x_train, y_train)
confidence = clf.score(x_test, y_test)
forecast_set = clf.predict(x_recent)

#Adding a forecast column to the dataframe
df['Forecast'] = np.nan

#############################################################
#VISUALIZING THE DATA
#############################################################
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()

one_day = 86400

next_unix = last_unix + one_day

#For each day's forecast, set values in dataframe
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

#Plotting
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()









