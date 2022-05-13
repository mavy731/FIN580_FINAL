import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def yh_get_data(ticker, t1, t2, t, h):
    df = yf.download(ticker, start = t1, end =t2 ,progress=False)
    data = df[['Adj Close']]
    data = data.rename(columns={'Adj Close':'Price'})
    data['Return'] = np.log(data['Price']) - np.log(data['Price'].shift(1))
    data['pos'] = data.Return.rolling(5).std()/data.Return.rolling(30).std()
    data['mom'] = data.Return.rolling(t).sum()
    data['sig'] = data.mom.apply(lambda x: 1 if x >= 0 else -1)
    df = data.dropna()
    df['r_y'] = df.Return.shift(1)
    pos = 0
    df['r'] = 0
    for i in range(len(df)):
        index = df.index[i]
        if (i % h == 0):
            pos = df.loc[index, 'pos'] * df.loc[index, 'sig']
            df.loc[index, 'r'] = pos * df.loc[index, 'r_y']
        else:
            df.loc[index, 'r'] = pos * df.loc[index, 'r_y']
    df = df.dropna()
    return df['r']

def get_max_drawdown_fast(array):
    drawdowns = []
    max_so_far = array[0]
    for i in range(len(array)):
        if array[i] > max_so_far:
            drawdown = 0
            drawdowns.append(drawdown)
            max_so_far = array[i]
        else:
            drawdown = max_so_far - array[i]
            drawdowns.append(drawdown)
    return max(drawdowns)


list = ['ES=F','YM=F','NQ=F','GC=F','SI=F','PL=F', 'HG=F','PA=F','CL=F',
        'HO=F','NG=F', 'ZC=F', 'ZO=F','ZS=F']
tick = list[8]
start_in = '2005-01-01'
end_in = '2015-12-31'
lookback =30
holding_period = 10
start_out = '2016-01-01'
end_out = '2021-12-31'

def port(list, t1, t2, t, h, cost):
    ret = []
    for f in list:
        df = yh_get_data(f, t1, t2, t, h)
        df = df.rename(str(t)+'-'+str(h))
        ret.append(df)
    all = pd.concat(ret, axis=1)
    all['Return'] = all.mean(axis=1)
    total_cost_approx = cost * len(all) / j
    all.Return = all.Return - total_cost_approx / len(all)
    print('return', np.mean(all.Return) * 250)
    print('sharpe', np.mean(all.Return) * 250 / (np.std(all.Return) * np.sqrt(250)))
    print('drawdown', get_max_drawdown_fast(np.array(all.Return.cumsum())))
    return all.Return
cost = 0.001
window =[]
for i in [5, 10, 20, 30, 60]:
    l = []
    for j in [5, 10, 20, 30]:
        print(i,j)
        df = port(list, start_in, end_in, i, j, cost)
        l.append(df)
    all = pd.concat(l, axis=1)
    fig = plt.figure(figsize=(20,8))
    plt.plot(all.cumsum())
    plt.legend(labels = all.columns)
    plt.savefig('Graph\Return_in_Sample'+str(i)+'.png')
    plt.title('Return in Sample'+str(i)+'-'+str(j))
    plt.grid(linestyle='-.')
    plt.show()
    window.append(all)
i=5
l = []
for j in [5, 10, 20, 30]:
    print(i, j)
    df = port(list, start_in, end_in, i, j, cost)
    l.append(df)
all = pd.concat(l, axis=1)
fig = plt.figure(figsize=(20,8))
plt.plot(all.cumsum())
plt.legend(labels = all.columns)
plt.savefig('Graph\Return_out_Sample'+str(i)+'.png')
plt.title('Return out Sample'+str(i)+'-'+str(j))
plt.grid(linestyle = '-.')
plt.show()

i=10
l = []
for j in [5, 10, 20, 30]:
    print(i, j)
    df = port(list, start_in, end_in, i, j, cost)
    l.append(df)
all = pd.concat(l, axis=1)
fig = plt.figure(figsize=(20,8))
plt.plot(all.cumsum())
plt.legend(labels = all.columns)
plt.savefig('Graph\Return_out_Sample'+str(i)+'.png')
plt.title('Return out Sample'+str(i)+'-'+str(j))
plt.grid(linestyle = '-.')
plt.show()














#
# fig = plt.figure(figsize=(20,8))
# plt.plot(df[['Return','r']].cumsum())
# plt.legend(labels = ['Return', 'r'])
# plt.show()
