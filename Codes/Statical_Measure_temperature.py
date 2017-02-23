import pandas as pd
import matplotlib
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import plotly.tools as tls
style.use("ggplot")
statistical_dataset = pd.read_csv(r"G:\Semester - 3\Project\Datasets\raw\meta\statistical_measure_temp.csv")
statistical_dataset.dropna(axis = 0, inplace = True)

plt.subplot(1,2,1)

time_values = statistical_dataset['end'] - statistical_dataset['start']
std_dev_measures = statistical_dataset['std_temp']
slope_measures = statistical_dataset['slope_temp']
time_labels_p1 = [str(time) for time in time_values[:9]]
time_labels_p2 = [str(time) for time in time_values[9:18]]
time_labels_p3 = [str(time) for time in time_values[18:27]]
time_labels_p4 = [str(time) for time in time_values[27:36]]
time_labels_p5 = [str(time) for time in time_values[36:45]]
time_labels_p6 = [str(time) for time in time_values[45:54]]
time_labels_p7 = [str(time) for time in time_values[54:63]]
questions = ["Q"+str(i)+"-" for i in range(1,10,1)]
questions_array = np.array(questions)
x_labels_p1 = np.core.defchararray.add(questions, time_labels_p1)
x_labels_p2 = np.core.defchararray.add(questions, time_labels_p2)
x_labels_p3 = np.core.defchararray.add(questions, time_labels_p3)
x_labels_p4 = np.core.defchararray.add(questions, time_labels_p4)
x_labels_p5 = np.core.defchararray.add(questions, time_labels_p5)
x_labels_p6 = np.core.defchararray.add(questions, time_labels_p6)
x_labels_p7 = np.core.defchararray.add(questions, time_labels_p7)

plt.bar(time_values[:9], slope_measures[:9], label = "P1", color = "b", width = 4)
plt.xticks(time_values[:9], x_labels_p1, rotation = 'vertical')
plt.ylabel("Slope")
plt.legend()


plt.subplot(1,2,2)
plt.bar(time_values[9:18], slope_measures[9:18], label = "P2", color = "r", width = 4)
plt.xticks(time_values[9:18], x_labels_p2, rotation = 'vertical')
plt.ylabel("Slope")
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.legend()
plt.show()


plt.figure()

plt.subplot(1,2,1)
plt.bar(time_values[18:27], slope_measures[18:27], label = "P3", color = "g", width = 4)
plt.xticks(time_values[18:27], x_labels_p3, rotation = 'vertical')
plt.legend()
plt.margins(0.2)
plt.ylabel("Slope")
plt.subplots_adjust(bottom=0.15)

plt.subplot(1,2,2)
plt.bar(time_values[27:36], slope_measures[27:36], label = "P4", color = "brown", width = 4)
plt.xticks(time_values[27:36], x_labels_p4, rotation = 'vertical')
plt.ylabel("Slope")
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)

plt.legend()
plt.show()


plt.figure()

plt.subplot(1,3,1)
plt.bar(time_values[36:45], slope_measures[36:45], label = "P5", color = "yellow", width = 4)
plt.xticks(time_values[36:45], x_labels_p3, rotation = 'vertical')
plt.legend()
plt.ylabel("Slope")
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)

plt.subplot(1,3,2)
plt.bar(time_values[45:54], slope_measures[45:54], label = "P6", color = "black", width = 4)
plt.xticks(time_values[45:54], x_labels_p4, rotation = 'vertical')
plt.legend()
plt.ylabel("Slope")
plt.subplot(1,3,3)
plt.bar(time_values[54:63], slope_measures[54:63], label = "P7", color = "orange", width = 4)
plt.xticks(time_values[54:63], x_labels_p4, rotation = 'vertical')
plt.ylabel("Slope")
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)

plt.legend()
plt.show()





##plt.xticks([time for time in time_values[:9]],
##           ['%i' %x for x in xrange(1,10,1)])




