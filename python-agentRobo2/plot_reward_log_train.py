# coding:utf-8
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import argparse
import numpy as np

df1 = pd.read_csv('cut_rewardAction3_fix.log')
df2 = pd.read_csv('cut_rewardAction3V_fix.log')
#df1 = pd.read_csv('cut_rewardAction3A.log')
#df2 = pd.read_csv('cut_rewardAction3VA.log')

x1 = df1.columns[0]
y1 = df1.columns[1]
x2 = df2.columns[0]
y2 = df2.columns[1]
#x3 = df3.columns[0]
#y3 = df3.columns[1]
#x4 = df4.columns[0]
#y4 = df4.columns[1]


df1[y1] = pd.rolling_mean(df1[y1], window=300)
df2[y2] = pd.rolling_mean(df2[y2], window=300)
#df3[y3] = pd.rolling_mean(df3[y3], window=500)
#df4[y4] = pd.rolling_mean(df4[y4], window=500)

y1_array = np.array(df1[y1])
y2_array = np.array(df2[y2])
#y3_array = np.array(df3[y3])
#y4_array = np.array(df4[y4])

print y1_array.max()
print y2_array.max()

fig, ax = plt.subplots(1, 1)
#plt.xticks(range(0,1000001,200000))
#plt.xticks(range(0,300001,50000))
plt.xticks(range(0,600001,100000))

plt.xlabel("Cycle") # x軸のラベル
plt.ylabel("Score") # y軸のラベル

'''
plt.plot(list(df1[x1])[:15001], list(df1[y1])[:15001], label='Model1', color='red', linewidth=2.5)
plt.plot(list(df2[x2])[:15001], list(df2[y2])[:15001], label='Model2', color='green', linewidth=2.5)
plt.plot(list(df3[x3])[:15001], list(df3[y3])[:15001], label='Model3', color='blue', linewidth=2.5)
plt.plot(list(df4[x4])[:15001], list(df4[y4])[:15001], label='Model4', color='#ffc700', linewidth=2.5)

plt.plot(list(df1[x1]), y1_array, label='Model1V', color='red', linewidth=2.5)
plt.plot(list(df2[x2]), y2_array, label='Model1V', color='green', linewidth=2.5)
plt.plot(list(df3[x3]), y3_array, label='Model3V', color='blue', linewidth=2.5)
plt.plot(list(df4[x4]), y4_array, label='Model4V', color='#ffc700', linewidth=2.5)


plt.plot(list(df1[x1]), y1_array, label='Model1_forward', color='red', linewidth=2.5)
plt.plot(list(df2[x2]), y2_array, label='Model1V_forward', color='green', linewidth=2.5)
'''
plt.plot(list(df1[x1]), y1_array, label='Model1A', color='red', linewidth=2.5)
plt.plot(list(df2[x2]), y2_array, label='Model1VA', color='green', linewidth=2.5)
plt.legend(loc = 'upper left') #これをしないと凡例出てこない(lower⇆upper, left⇆ center ⇆right)
plt.show()

'''
df = pd.read_csv('cut_rewardAction3A.log')
c = df.columns[0]
s = df.columns[1]
e = df.columns[2]
g = df.columns[3]

cycle = np.array(df[c])[:10000]
score = np.array(df[s])[:10000].astype(float)
episode = np.array(df[e])[:10000]
goal_time = np.array(df[g])[:10000]

np.average(score[501:1000]-2)

score[0:1000] = score[0:1000] -1

score[1000:3000] = score[1000:3000] -1

with open("hoge.log", 'w') as the_file:
                the_file.write('Cycle,Score,Episode,GoalTime \n')
for i in range(goal_time.shape[0]):
    with open("hoge.log", 'a') as the_file:
        the_file.write(str(cycle[i]) +
       ',' + str(score[i]) +
       ',' + str(episode[i]) +
       ',' + str(goal_time[i]) + '\n')
'''
