import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def yh_get_data(ticker, t1, t2):
    df = yf.download(ticker, start = t1, end = t2 ,progress=False)
    data = df[['Adj Close']]
    data = data.rename(columns={'Adj Close':'Price'})
    data['Return'] = np.log(data['Price']) - np.log(data['Price'].shift(1))
    return data

def cal_mom(df):
    df['mom_5'] = df.Return.rolling(5).sum()
    df['mom_30'] = df.Return.rolling(30).sum()
    df['mom_60'] = df.Return.rolling(60).sum()
    df['mom_120'] = df.Return.rolling(120).sum()
    df['sig_5'] = df.mom_5.apply(lambda x: 1 if x>=0 else -1)
    df['sig_30'] = df.mom_30.apply(lambda x: 1 if x >= 0 else -1)
    df['sig_60'] = df.mom_60.apply(lambda x: 1 if x >= 0 else -1)
    df['sig_120'] = df.mom_120.apply(lambda x: 1 if x >= 0 else -1)
    df = df.dropna()
    return df

def cal_ret(df):
    df['r_y'] = df.Return.shift(1)
    df['r_5'] = df.sig_5 * df.r_y
    df['r_30'] = df.sig_30 * df.r_y
    df['r_60'] = df.sig_60 * df.r_y
    df['r_120'] = df.sig_120 * df.r_y
    df = df.dropna()
    return df[['Return','r_5','r_30','r_60','r_120']]

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

def num_of_trade(array):
    count = 0
    for i in range(len(array)-1):
        if(array[i]*array[i+1]<0):
            count = count+1
    return count


tick = '^GSPC'
start = '2010-01-01'
end = '2021-12-31'

sp = yh_get_data('^GSPC', start, end)
df = cal_mom(sp)
ret = cal_ret(df)

fig,ax1 = plt.subplots(figsize=(20,8))
ax1.plot(ret[['Return','r_5','r_30','r_60','r_120']].cumsum())
ax1.legend(labels = ['Return','r_5','r_30','r_60','r_120'])
plt.title('Window Selection Test')
plt.grid(linestyle = '-.')
plt.savefig('Graph\window selection_full.png')
plt.show()

# tick = '^GSPC'
# start = '2018-01-01'
# end = '2021-12-31'
#
# sp = yh_get_data('^GSPC', start, end)
# df = cal_mom(sp)
# ret = cal_ret(df)
#
# fig,ax1 = plt.subplots(figsize=(20,8))
# ax1.plot(ret[['Return','r_5','r_30','r_60','r_120']].cumsum())
# ax1.legend(labels = ['Return','r_5','r_30','r_60','r_120'])
# plt.title('Window Selection Test 2018-2021')
# plt.grid(linestyle = '-.')
# plt.savefig('Graph\window selection_1821.png')
# plt.show()
#
# tick = '^GSPC'
# start = '2014-01-01'
# end = '2017-12-31'
#
# sp = yh_get_data('^GSPC', start, end)
# df = cal_mom(sp)
# ret = cal_ret(df)
#
# fig,ax1 = plt.subplots(figsize=(20,8))
# ax1.plot(ret[['Return','r_5','r_30','r_60','r_120']].cumsum())
# ax1.legend(labels = ['Return','r_5','r_30','r_60','r_120'])
# plt.title('Window Selection Test 2014-2017')
# plt.grid(linestyle = '-.')
# plt.savefig('Graph\window selection_1417.png')
# plt.show()
#
# tick = '^GSPC'
# start = '2009-06-01'
# end = '2013-12-31'
#
# sp = yh_get_data('^GSPC', start, end)
# df = cal_mom(sp)
# ret = cal_ret(df)
#
# fig,ax1 = plt.subplots(figsize=(20,8))
# ax1.plot(ret[['Return','r_5','r_30','r_60','r_120']].cumsum())
# ax1.legend(labels = ['Return','r_5','r_30','r_60','r_120'])
# plt.title('Window Selection Test 2010-2013')
# plt.grid(linestyle = '-.')
# plt.savefig('Graph\window selection_1013.png')
# plt.show()
print(df)
list = ['Return','r_5','r_30','r_60','r_120']
for x in list:
    print(x)
    print('return', np.sum(ret[x]))
    print('sharpe', np.mean(ret[x])*250/ (np.std(ret[x])*np.sqrt(250)))
    print('drawdown', get_max_drawdown_fast(np.array(ret[x].cumsum())))

l = ['5','30','60','120']
for y in l:
    print(y)
    x = 'sig_'+y
    r = 'r_'+y
    print('trades', num_of_trade(np.array(df[x])))
    print('return per trade', np.sum(ret[r])/num_of_trade(np.array(df[x])))