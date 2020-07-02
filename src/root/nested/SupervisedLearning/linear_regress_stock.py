import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

plt.rc('figure', figsize=(12,8.0))

# create a 3 -day trend of the price
# if the price today is greater than the price yesterday, assign +1
# if the price today is less than the price yesterday, assign -1
# then, if the majority of the values in the past 3 days are +1,
# then assign the trend as positive, else assign, the trend as negative

# the rest below assumes that you have a dataframe with the stock price data
df.dropna(inplace=True)
df.plot(x='date', y='close')
# the above assumes that you have columns in the dataframe with 'date'
# and 'close' labels.

# for a subset of the data, we will plot our trend
start_date = '2018-06-01'
end_date = '2018-07-31'
plt.plot(
    'date', 'close', 'k--',
    data = (
        df.loc[pd.to_datetime(df.date).between(start_date, end_date)]
    )
)

# the below requires that in df, you have a column named 'trend_3_day'
# this contains the 3 day trend value of -1, or 1.

plt.scatter(
    'date', 'close', color='b', label='pos trend',
    data = (
        df.loc[df.trend_3_day == 1 & pd.to_datetime(df.date).between(start_date, end_date)]
    )
)

plt.scatter(
    'date', 'close', color='r', label='neg trend',
    data = (
        df.loc[(df.trend_3_day == -1) & pd.to_datetime(df.date).between(start_date, end_date)]
    )
)

plt.legend()
plt.xticks(rotation = 90);

# the below is the actual creating of the linear regression model.
# again, this requires that there is a column in the df called day_prev_close
# this is simply a 1-lag on the close price
features = ['day_prev_close', 'trend_3_day']
target = 'close'
# splitting the data into train and test partitions
X_train, X_test = df.loc[:2000, features], df.loc[2000:, features]
y_train, y_test = df.loc[:2000, target], df.loc[2000:, target]
# Create linear regression object. Don't include an intercept,
regr = linear_model.LinearRegression(fit_intercept=False)
# Train the model using the training set
regr.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = regr.predict(X_test)
# Print the root mean squared error of your predictions
print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

# the RMSE above should be lower than if we use the NAIF prediction,
# which is simply predicting yesterday's price as today's price.

# Print the variance score (1 is perfect prediction)
print('Variance Score: {0:.2f}'.format(r2_score(y_test, y_pred)))
# Plot the predicted values against their corresponding true values
plt.scatter(y_test, y_pred)
plt.plot([140, 240], [140, 240], 'r--', label='perfect fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend();


# there is no predictive quality to this model, but, what seems to have
# some possibility of success is the following:
#   1. Look at the predicted close price vs. actual close price.
#   2. If it is far above the actual close price, BUY.
#   3. If it is far below the actual close price, SELL.
#   4. How to determine "far below" or "far above"?
#   4. Probably look at percentile of the "error".