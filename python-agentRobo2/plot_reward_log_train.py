# coding:utf-8
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import argparse
import numpy as np


df1 = pd.read_csv('cut_rewardAction3V.log')
df2 = pd.read_csv('cut_rewardAction5V.log')
df3 = pd.read_csv('cut_rewardAction7_1V.log')
df4 = pd.read_csv('cut_rewardAction7_2V.log')

x1 = df1.columns[0]
y1 = df1.columns[1]
x2 = df2.columns[0]
y2 = df2.columns[1]
x3 = df3.columns[0]
y3 = df3.columns[1]
x4 = df4.columns[0]
y4 = df4.columns[1]

np.array(df1[y1]).max()
np.array(df2[y2]).max()
np.array(df3[y3]).max()
np.array(df4[y4]).max()

df1[y1] = pd.rolling_mean(df1[y1], window=500)
df2[y2] = pd.rolling_mean(df2[y2], window=500)
df3[y3] = pd.rolling_mean(df3[y3], window=500)
df4[y4] = pd.rolling_mean(df4[y4], window=500)

fig, ax = plt.subplots(1, 1)
#plt.xticks(range(0,15001,2000))
plt.xticks(range(0,1000001,200000))
plt.xlabel("Cycle") # x軸のラベル
plt.ylabel("Score") # y軸のラベル
#plt.xlim(-1, 7) # xを-0.5-7.5の範囲に限定

'''
plt.plot(list(df1[x1])[:15001], list(df1[y1])[:15001], label='Model1', color='red', linewidth=2.5)
plt.plot(list(df2[x2])[:15001], list(df2[y2])[:15001], label='Model2', color='green', linewidth=2.5)
plt.plot(list(df3[x3])[:15001], list(df3[y3])[:15001], label='Model3', color='blue', linewidth=2.5)
plt.plot(list(df4[x4])[:15001], list(df4[y4])[:15001], label='Model4', color='#ffc700', linewidth=2.5)
'''
plt.plot(list(df1[x1]), list(df1[y1]), label='Model1', color='red', linewidth=2.5)
plt.plot(list(df2[x2]), list(df2[y2]), label='Model2', color='green', linewidth=2.5)
plt.plot(list(df3[x3]), list(df3[y3]), label='Model3', color='blue', linewidth=2.5)
plt.plot(list(df4[x4]), list(df4[y4]), label='Model4', color='#ffc700', linewidth=2.5)
plt.legend(loc = 'upper right') #これをしないと凡例出てこない(lower⇆upper, left⇆ center ⇆right)
plt.show()
