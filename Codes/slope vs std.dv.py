import pandas as pd
import matplotlib
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import plotly.tools as tls

fig, ax = plt.subplots()
s = 121
style.use("fivethirtyeight")
statistical_dataset = pd.read_csv(r"G:\Semester - 3\Project\Datasets\raw\meta\std_dev_slope_TeamAB.csv")
statistical_dataset.dropna(axis = 0, inplace = True)

std1 = statistical_dataset['std_temp'][:9]
slope1 = statistical_dataset['slope_temp'][:9]
std2 = statistical_dataset['std_temp'][9:19]
slope2 = statistical_dataset['slope_temp'][9:19]
std3 = statistical_dataset['std_temp'][19:28]
slope3 = statistical_dataset['slope_temp'][19:28]
std4 = statistical_dataset['std_temp'][28:38]
slope4 = statistical_dataset['slope_temp'][28:38]
std5 = statistical_dataset['std_temp'][38:48]
slope5 = statistical_dataset['slope_temp'][38:48]
std6 = statistical_dataset['std_temp'][48:57]
slope6 = statistical_dataset['slope_temp'][48:57]
std7 = statistical_dataset['std_temp'][57:66]
slope7 = statistical_dataset['slope_temp'][57:66]
std8 = statistical_dataset['std_temp'][66:76]
slope8 = statistical_dataset['slope_temp'][66:76]
std9 = statistical_dataset['std_temp'][76:86]
slope9 = statistical_dataset['slope_temp'][76:86]
std10 = statistical_dataset['std_temp'][86:95]
slope10 = statistical_dataset['slope_temp'][86:95]
std11 = statistical_dataset['std_temp'][95:104]
slope11 = statistical_dataset['slope_temp'][95:104]
std12 = statistical_dataset['std_temp'][104:113]
slope12 = statistical_dataset['slope_temp'][104:113]
std13 = statistical_dataset['std_temp'][113:123]
slope13 = statistical_dataset['slope_temp'][113:123]
std14 = statistical_dataset['std_temp'][123:]
slope14 = statistical_dataset['slope_temp'][123:]

ax.scatter(std1, slope1, color='r',  s=4*s, marker='o', alpha=.8, label = "p1")
ax.scatter(std2, slope2, color='g',  s=4*s, marker='^', alpha=.8, label = "p2")
ax.scatter(std3, slope3, color="pink",  s=4*s, marker='v', alpha=.8, label = "p3")
ax.scatter(std4, slope4, color='k',  s=4*s, marker='<', alpha=.8, label = "p4")
ax.scatter(std5, slope5, color='c',  s=4*s, marker='>', alpha=.8, label = "p5")
ax.scatter(std6, slope6, color='m',  s=4*s, marker='8', alpha=.8, label = "p6")
ax.scatter(std7, slope7, color='y',  s=4*s, marker='s', alpha=.8, label = "p7")
ax.scatter(std8, slope8, color='brown',  s=4*s, marker='p', alpha=.8, label = "p8")
ax.scatter(std9, slope9, color='purple',  s=4*s, marker='*', alpha=.8, label = "p9")
ax.scatter(std10, slope10, color='orange',  s=4*s, marker='h', alpha=.8, label = "p10")
ax.scatter(std11, slope11, color='black',  s=4*s, marker='+', alpha=.8, label = "p11")
ax.scatter(std12, slope12, color='red',  s=4*s, marker='x', alpha=.8, label = "p12")
ax.scatter(std13, slope13, color='green',  s=4*s, marker='D', alpha=.8, label = "p13")
ax.scatter(std14, slope14, color='blue',  s=4*s, marker=',', alpha=.5, label = "p14")

plt.legend()
plt.show()

