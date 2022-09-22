
#Imports
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from math import sqrt
from sklearn.metrics import mean_squared_error
from pandas import DataFrames

df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
print('****************************************************')
print(df.head())


print('****************************************************')






# Here I start cleaning the data. Firstly, converting Timestamp to datetime64
df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')
# sets the index as the date
df.index = df.Timestamp
# Resamples the data to the average daily value of each column. Removes excessive frequency
df = df.resample('D').mean()
# drops any missing values that are present
df = df.dropna()

print('****************************************************')
print(df.head())





print('****************************************************')
print(df.shape)





# graph bitcoin price over the years
df.Weighted_Price.plot(title = "Bitcoin Price", figsize=(14,6))
plt.tight_layout()
plt.xlabel('Years')
plt.ylabel('US Dollars')
plt.show()
# As the graph shows 2017-2021 price behavior looks signficantly different than 2012-2017





autocorrelation_plot(df)
plt.show()





# let's look at the past 200 days to possibly adjust our data to this period
df.Weighted_Price.iloc[-200:].plot(title = "Bitcoin Price", figsize=(14,6))
plt.tight_layout()
plt.xlabel('Years')
plt.ylabel('US Dollars')
plt.show()





# Since the first couple years of bitcoin don't properly represent the movement and volatility of the price -
# I decide to simply focus on the previous 4 years of data from March 31st, 2021.
df2 = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
df2.Timestamp = pd.to_datetime(df2.Timestamp, unit='s')
df2.index = df2.Timestamp
df2 = df2.resample('D').mean()
df2 = df2.dropna()
# changes data to strictly to the previous 4 years, which is March 2017 to March 2021
df2 = df2.iloc[(-365*4):]
print(df2.shape)





df2.Weighted_Price.plot(title = "Bitcoin Price", figsize=(14,6))
plt.tight_layout()
plt.xlabel('Dates')
plt.ylabel('$ Price')
plt.show()
# This data looks much more relevant for training a model. 
# However the recent spike will be involved in the testing data split - a tough prediction.


# # ARIMA




from statsmodels.tsa.arima_model import ARIMA
c1=0.94
c2=0.96
import statsmodels.api as sm
# We're going to create a dataframe for just the price (the index is still the date)
price = df2.Weighted_Price
# Next we're going to assign 70% percent of the data to training and 30% for testing
X = price.values
size = int(len(X) * 0.7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()





# walk-forward validation
for t in range(len(test)):
	model = sm.tsa.arima.ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()





# if we look at this model, the predicted is indistinuishable from the actual price
# this is simply because it's predicting day by day.
plt.figure(figsize=(15,8))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.xlabel('Days')
plt.ylabel('$ Price')
plt.title('Predicted vs. Actual BTC Price')
plt.show()





# I plot 50 days to more accurately see how the models works with its lag
plt.figure(figsize=(15,8))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test[-50:])
plt.plot(predictions[-50:], color='red')
plt.xlabel('Days')
plt.ylabel('$ Price')
plt.title('Predicted vs. Expected BTC Price Forecast')
plt.show()


# # Hybrid svr arima




from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
# method to be used later
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i-interval]
        diff.append(value)
    return np.array(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]







# Importing the dataset
df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')



df.index = df.Timestamp

print('****************************************************')
print(df)






df_day = df.resample('D').mean()

df_day = df.resample('D').mean()

# Find missing values
for i, r in df_day.iterrows():
    if r.isnull().sum() > 0:
        print(i)
        print(r)
        print()

# Print the records of the date between the missing records
print('Data at 2013-01-05')
print(df_day.loc['2013-01-05'])

print()
print('Data at 2013-01-09')
print(df_day.loc['2013-01-09'])


missing_replacement = df['2013-01-05': '2013-01-09'].mean(numeric_only=True)
df_day.loc['2013-01-06'] = missing_replacement
df_day.loc['2013-01-07'] = missing_replacement
df_day.loc['2013-01-08'] = missing_replacement

# Any missing value?
df_day.isnull().sum()





from datetime import date

window_size = (date(2020, 9, 14) - date(2020, 5, 10)).days

df_train = df_day.loc['2011-12-31':'2020-05-10']
df_test = df_day.loc['2013-05-11':]

df_train['Predicted_Price'] = df_train[['Weighted_Price']].shift(-window_size)

X_train = df_train[['Weighted_Price']].values[:-window_size].reshape(-1,1)
y_train = df_train.Predicted_Price.values[:-window_size]

y_test = df_test.Weighted_Price.values

y_test, y_test.shape





from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=1000)
svr.fit(X_train, y_train)
print(X_train.shape)
print(y_train.shape)
print(y_test.shape)





print(len(test))





# Split the data as usual 70, 30
price = df2.Weighted_Price
X = price.values
datesX = price.index
size = int(len(X) * 0.70)
train, test = X[0:size], X[size:len(X)]
days_in_year = 365
plotDates = datesX[size:len(X)]

# Next we will forecast with ARIMA using 5,1,0
differenced = difference(train, days_in_year)
model = sm.tsa.arima.ARIMA(differenced, order=(5, 1, 0))
model_fit = model.fit()
start_index = len(differenced)
end_index = start_index + 438
mergedmodel,c1,c2=DataFrames.merge(svr,model_fit)
forecast = mergedmodel.predict(start=start_index, end=end_index)

history = [x for x in train]
day = 1
predicted_results = list()

# store predicted results 
for yhat in forecast:
    inverted = inverse_difference(history, yhat, days_in_year)
    print("Predicted Day %d: %f" % (day, inverted))
    history.append(inverted)
    predicted_results.append(inverted)
    day += 1





print(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())










