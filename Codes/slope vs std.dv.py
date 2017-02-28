import pandas as pd
import matplotlib
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import plotly.tools as tls

style.use("fivethirtyeight")
fig, ax = plt.subplots()
s = 121
statistical_dataset = pd.read_csv(r"G:\Semester - 3\Project\Datasets\raw\meta\std_dev_slope_TeamAB.csv")
statistical_dataset.dropna(axis = 0, inplace = True)
plt.title("SLOPE VS STD.DEV - Comparison with scores")
plt.xlabel("Standard Deviation")
plt.ylabel("Slope")

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

ax.scatter(std1, slope1, color='r',  s=2*s, marker='o', alpha=.8, label = "P1(shape), SCORE-4(color)")
ax.scatter(std2, slope2, color='g',  s=2*s, marker='^', alpha=.8, label = "P2(shape), SCORE-3(color)")
ax.scatter(std3, slope3, color="r",  s=2*s, marker='v', alpha=.8, label = "P3(shape), SCORE-4(color)")
ax.scatter(std4, slope4, color='b',  s=2*s, marker='<', alpha=.4, label = "P4(shape), SCORE-1(color)")
ax.scatter(std5, slope5, color='b',  s=2*s, marker='>', alpha=.4, label = "P5(shape), SCORE-1(color)")
ax.scatter(std6, slope6, color='b',  s=2*s, marker='8', alpha=.4, label = "P6(shape), SCORE-1(color)")
ax.scatter(std7, slope7, color='g',  s=2*s, marker='s', alpha=.8, label = "P7(shape), SCORE-3(color)")
ax.scatter(std8, slope8, color='brown',  s=2*s, marker='p', alpha=.8, label = "P8(shape), SCORE-0(color)")
ax.scatter(std9, slope9, color='b',  s=2*s, marker='*', alpha=.4, label = "P9(shape), SCORE-1(color)")
ax.scatter(std10, slope10, color='brown',  s=2*s, marker='h', alpha=.8, label = "P10(shape), SCORE-0(color)")
ax.scatter(std11, slope11, color='brown',  s=2*s, marker='+', alpha=.8, label = "P11(shape), SCORE-0(color)")
ax.scatter(std12, slope12, color='b',  s=2*s, marker='x', alpha=.4, label = "P12(shape), SCORE-1(color)")
ax.scatter(std13, slope13, color='b',  s=2*s, marker='D', alpha=.4, label = "P13(shape), SCORE-1(color)")
ax.scatter(std14, slope14, color='b',  s=2*s ,marker='d', alpha=.4, label = "P14(shape), SCORE-1(color)")
plt.legend()

fig, ax1 = plt.subplots()
plt.title("SLOPE VS STD.DEV - Comparison with self-confidences")
plt.xlabel("Standard Deviation")
plt.ylabel("Slope")


ax1.scatter(std1, slope1, color='r',  s=2*s, marker='o', alpha=.8, label = "P1(shape), conf-high(color)")
ax1.scatter(std2, slope2, color='r',  s=2*s, marker='^', alpha=.8, label = "p2(shape), conf-high(color)")
ax1.scatter(std3, slope3, color='k',  s=2*s, marker='v', alpha=.8, label = "P3(shape), conf-very low(color)")
ax1.scatter(std4, slope4, color='g',  s=2*s, marker='<', alpha=.8, label = "P4(shape), conf-medium(color)")
ax1.scatter(std5, slope5, color='b',  s=2*s, marker='>', alpha=.8, label = "P5(shape), conf-low(color)")
ax1.scatter(std6, slope6, color='g',  s=2*s, marker='8', alpha=.8, label = "p6(shape), conf-medium(color)")
ax1.scatter(std7, slope7, color='r',  s=2*s, marker='s', alpha=.8, label = "P7(shape), conf-high(color)")
ax1.scatter(std8, slope8, color='b',  s=2*s, marker='p', alpha=.8, label = "P8(shape), conf-low(color)")
ax1.scatter(std9, slope9, color='g',  s=2*s, marker='*', alpha=.8, label = "P9(shape), conf-medium(color)")
ax1.scatter(std10, slope10, color='b',  s=2*s, marker='h', alpha=.8, label = "P10(shape), conf-low(color)")
ax1.scatter(std11, slope11, color='r',  s=2*s, marker='+', alpha=.8, label = "P11(shape), conf-high(color)")
ax1.scatter(std12, slope12, color='g',  s=2*s, marker='x', alpha=.8, label = "P12(shape), conf-medium(color)")
ax1.scatter(std13, slope13, color='m',  s=2*s, marker='D', alpha=.8, label = "P13(shape), conf-very high(color)")
ax1.scatter(std14, slope14, color='k',  s=2*s, marker=',', alpha=.4, label = "P14(shape), conf-very low(color)")

plt.legend()
plt.show()

