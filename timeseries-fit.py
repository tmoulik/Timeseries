#!/usr/bin/python
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm 
from datetime import datetime
import datetime
from dateutil.relativedelta import relativedelta
#tda = pd.read_excel("project1.xlsx")
tda = pd.read_excel("../sample-data.xlsx")
tda
#ts = tda[0:50]
ts = tda
col2 = 'Price'
col1 = 'Date'
ts[col2] = ts[col2].astype('float64')
ts[col2] = pd.to_numeric(ts[col2])
ts = ts.drop_duplicates(subset=col1)
ts = ts.set_index(col1)

#Take Log of ts
ts_log = ts
ts_log[col2] = np.log(ts[col2])

# Comments : TS is fairly stationary.
nl = 6
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts_log_diff, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts_log_diff, ax=ax2)

model = ARIMA(ts_log, order=(3,1,0))
results_AR = model.fit()
val = results_AR.fittedvalues
diff2 = (val-ts_log_diff[col2])**2
#plt.plot(ts_log_diff)
#plt.plot(results_AR.fittedvalues, color='red')
#plt.title('RSS: %.4f' % sum(diff2))

#Scale to Actual Values.
predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

baseval=np.log(tda.ix[0]['Price'])
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=baseval)
predictions_ARIMA_log.head()

predictions_ARIMA_log.dropna(inplace=True)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
ts[col2] = np.exp(ts_log[col2])
new_data  = ts.iloc[1:]
#plt.plot(new_data)
#plt.plot(predictions_ARIMA)
diff2 = sum((predictions_ARIMA - new_data[col2])**2)
diff = np.sqrt(diff2)/len(new_data)
#plt.title('RMSE: %.4f'% diff)

# Predict some Values and Plot
startn = 10
endn= len(ts_log_diff)
ts_log_diff['forecast'] = results_AR.predict(start = startn, end= endn)
ts_log_diff[['Price', 'forecast']].plot()

# Predict Out of sample dates
#Get the last date
lastdate = ts_log_diff.index[len(ts_log_diff)-1]
newn=10
rng = pd.date_range(lastdate, periods=newn, freq='D')
future = pd.DataFrame(index=rng, columns= ts_log_diff.columns)
future=future[1:]
newts_log_diff = pd.concat([ts_log_diff, future])


startn2 = endn-2
endn2 = endn+newn
newts_log_diff[startn2:endn2].forecast = results_AR.predict(start=startn2,end=endn2)
newts_log_diff[['Price', 'forecast']].plot()