import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

#data analysis

start = '2018-01-01'
end = '2021-12-31'

def yh_get_data(ticker, t1, t2):
    df = yf.download(ticker, start = start, end =end ,progress=False)
    data = df[['Adj Close', 'Volume']]
    data = data.rename(columns={'Adj Close':'Price'})
    data['Return'] = np.log(data['Price']) - np.log(data['Price'].shift(1))
    return data

def cal_mom(df):
    df['mav_5'] = df.Price.rolling(5).mean()
    df['mav_30'] = df.Price.rolling(30).mean()
    df['mav_120'] = df.Price.rolling(120).mean()
    df['dif1'] =df.mav_5 - df.mav_30
    df['sig1'] = df.dif1/df.dif1.shift(1)
    df['dif2'] = df.mav_30 - df.mav_120
    df['sig2'] = df.dif2 / df.dif2.shift(1)
    df = df.dropna()
    return df
#ZC=F

sp500 = yh_get_data('^GSPC', start, end)
sp500_mom = cal_mom(sp500)

fig,ax1 = plt.subplots(figsize=(20,8))
ax2 = ax1.twinx()
ax1.plot(sp500_mom.index, sp500_mom[['mav_5','mav_30', 'mav_120']], )
ax2.plot(sp500_mom['Price'], color = 'black',alpha = 0.5)
ax1.legend(labels = ['mav_5','mav_30', 'mav_120'])
ax2.legend(labels = ['SP500 Price'], loc = 1)
plt.title('SP500 and MAV')
plt.grid(linestyle = '-.')
plt.savefig('Graph\SP500 and MAV.png')
plt.show()

start = '2010-01-01'
end = '2021-12-31'
sp500 = yh_get_data('^GSPC', start, end)
sp500_mom = cal_mom(sp500)
sp500_mom['pos'] = 0
sp500_mom['value'] = 0
print(sp500_mom)
df = sp500_mom

# value = 100000
# pos = 0
# for i in range(len(sp500_mom)):
#     index = sp500_mom.index[i]
#     p = sp500_mom.loc[index,'Price']
#     if(sp500_mom.loc[index,'sig1'] <0):
#         if (pos == 0):
#             sp500_mom.loc[index,'value'] = value
#             pos = value/p
#             sp500_mom.loc[index,'pos'] = pos
#         else:
#             value = pos*p
#             pos = 0
#             sp500_mom.loc[index, 'pos'] = pos
#             sp500_mom.loc[index, 'value'] = value
#     elif(pos == 0):
#         sp500_mom.loc[index, 'value'] = value
#     else:
#         sp500_mom.loc[index, 'value'] = pos*p
#         value = pos*p
#
# df = sp500_mom
# df['r'] = np.log(df.value) - np.log(df.value.shift(1))
# fig,ax1 = plt.subplots(figsize=(20,8))
# ax2 = ax1.twinx()
# ax1.plot(df['r'].cumsum(),color = 'r')
# ax2.plot(df['Price'],alpha = 0.5, color ='black')
# ax1.legend(labels = ['Return'])
# ax2.legend(labels = ['SP500 Price'], loc = 1)
# plt.title('Trading Return')
# plt.grid(linestyle = '-.')
# plt.savefig('Graph\SP500_simple_return1.png')
# plt.show()
# print(np.sum(df['r'])*360/len(df))
# print(np.std(df['r'])*np.sqrt(360))
# print((np.sum(df['r'])*360/len(df))/(np.std(df['r'])*np.sqrt(360)))

# value = 100000
# pos = 0
# for i in range(len(sp500_mom)):
#     index = sp500_mom.index[i]
#     p = sp500_mom.loc[index,'Price']
#     if(sp500_mom.loc[index,'sig2'] <0):
#         if (pos == 0):
#             sp500_mom.loc[index,'value'] = value
#             pos = value/p
#             sp500_mom.loc[index,'pos'] = pos
#         else:
#             value = pos*p
#             pos = 0
#             sp500_mom.loc[index, 'pos'] = pos
#             sp500_mom.loc[index, 'value'] = value
#     elif(pos == 0):
#         sp500_mom.loc[index, 'value'] = value
#     else:
#         sp500_mom.loc[index, 'value'] = pos*p
#         value = pos*p
#
# df = sp500_mom
# df['r'] = np.log(df.value) - np.log(df.value.shift(1))
# fig,ax1 = plt.subplots(figsize=(20,8))
# ax2 = ax1.twinx()
# ax1.plot(df['r'].cumsum(),color = 'g')
# ax2.plot(df['Price'],alpha = 0.5, color ='black')
# ax1.legend(labels = ['Return'])
# ax2.legend(labels = ['SP500 Price'], loc = 1)
# plt.title('Trading Return')
# plt.grid(linestyle = '-.')
# plt.savefig('Graph\SP500_simple_return2.png')
# plt.show()
# print(np.sum(df['r'])*360/len(df))
# print(np.std(df['r'])*np.sqrt(360))
# print((np.sum(df['r'])*360/len(df))/(np.std(df['r'])*np.sqrt(360)))