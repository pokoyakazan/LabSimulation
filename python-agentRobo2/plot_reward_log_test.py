# coding:utf-8
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import argparse
import numpy as np

df1 = pd.read_csv('testAction3.csv')
df2 = pd.read_csv('testAction5.csv')
df3 = pd.read_csv('testAction7_1.csv')
df4 = pd.read_csv('testAction7_2.csv')

#df1 = pd.read_csv('testAction3V.csv')
#df2 = pd.read_csv('testAction5V.csv')
#df3 = pd.read_csv('testAction7_1V.csv')
#df4 = pd.read_csv('testAction7_2V.csv')

x1 = df1.columns[2]
y1 = df1.columns[1]
x2 = df2.columns[2]
y2 = df2.columns[1]
x3 = df3.columns[2]
y3 = df3.columns[1]
x4 = df4.columns[2]
y4 = df4.columns[1]

score1 = np.array(df1[y1])[:1000]
score2 = np.array(df2[y2])[:1000]
score3 = np.array(df3[y3])[:1000]
score4 = np.array(df4[y4])[:1000]

score1 = score1.reshape((100,10))
score2 = score1.reshape((100,10))
score3 = score1.reshape((100,10))
score4 = score1.reshape((100,10))

'''
clear_count1 = []
perfect_model1 = []
average_time1 = []
min_time1 = []

clear_count2 = []
perfect_model2 = []
average_time2 = []
min_time2 = []

clear_count3 = []
perfect_model3 = []
average_time3 = []
min_time3 = []

clear_count4 = []
perfect_model4 = []
average_time4 = []
min_time4 = []
'''

with open("goalTime.csv", 'a') as the_file:
    the_file.write("Acion3,,\nModel,AverageTime,FastestTime\n")
for i in range(score1.shape[0]):
    clear_count = np.sum(score1[i] > 0)
    clear_count1.append(clear_count)
    if(clear_count==10):
        #perfect_model1.append((i+1)*10000)
        #perfect_model1.append(i)
        #average_time1.append(np.average(score1[i]))
        #min_time1.append(np.average(score1[i]))
        #logファイルへの書き込み
        with open("goalTime.csv", 'a') as the_file:
            the_file.write(str((i+1)*10000) +
                       ',' + str(np.average(score1[i])) +
                       ',' + str(np.min(score1[i])) + '\n')

with open("goalTime.csv", 'a') as the_file:
    the_file.write("Acion5,,\nModel,AverageTime,FastestTime\n")
for i in range(score2.shape[0]):
    clear_count = np.sum(score2[i] > 0)
    clear_count2.append(clear_count)
    if(clear_count==10):
        with open("goalTime.csv", 'a') as the_file:
            the_file.write(str((i+1)*10000) +
                       ',' + str(np.average(score2[i])) +
                       ',' + str(np.min(score2[i])) + '\n')

with open("goalTime.csv", 'a') as the_file:
    the_file.write("Acion7_1,,\nModel,AverageTime,FastestTime\n")
for i in range(score3.shape[0]):
    clear_count = np.sum(score3[i] > 0)
    clear_count3.append(clear_count)
    if(clear_count==10):
        with open("goalTime.csv", 'a') as the_file:
            the_file.write(str((i+1)*10000) +
                       ',' + str(np.average(score3[i])) +
                       ',' + str(np.min(score3[i])) + '\n')

with open("goalTime.csv", 'a') as the_file:
    the_file.write("Acion7_2,,\nModel,AverageTime,FastestTime\n")
for i in range(score4.shape[0]):
    clear_count = np.sum(score4[i] > 0)
    clear_count4.append(clear_count)
    if(clear_count==10):
        the_file.write(str((i+1)*10000) +
                   ',' + str(np.average(score4[i])) +
                   ',' + str(np.min(score4[i])) + '\n')


plt.xticks(range(0,1000001,200000))
plt.xlabel("Cycle") # x軸のラベル
plt.ylabel("Score") # y軸のラベル
#plt.xlim(-1, 7) # xを-0.5-7.5の範囲に限定

x_axis = range(10000,1000001,10000)

plt.plot(x_axis, clear_count1, label='Model1', color='red', linewidth=2.5)
plt.plot(x_axis, clear_count2, label='Model2', color='green', linewidth=2.5)
plt.plot(x_axis, clear_count3, label='Model3', color='blue', linewidth=2.5)
plt.plot(x_axis, clear_count4, label='Model4', color='#ffc700', linewidth=2.5)
plt.legend(loc = 'upper right') #これをしないと凡例出てこない(lower⇆upper, left⇆ center ⇆right)
plt.show()
