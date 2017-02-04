import pandas as pd
import matplotlib
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import plotly.tools as tls
style.use("fivethirtyeight")
fig = plt.figure()

gridnumber = range(1,4)
score_dataset = pd.read_csv(r"G:\Semester - 3\Project\Datasets\raw\meta\scores.csv")
score_dataset.dropna(axis = 0, inplace = True)
#print type(score_dataset)
x_labels = ["P1", "P2","P3", "P4","P5", "P6","P7", "P8","P9", "P10","P11", "P12","P13", "P14"]
scores = score_dataset['scores']
grades = score_dataset['Grade']
fig.suptitle('Tally between Temperatures and the score, grades',  fontsize=30)
ax = plt.subplot(122)
scores.plot(kind='bar', title ="Scores-of-Participants",figsize=(10,5),legend=True,color = "blue", fontsize=12, bottom = grades)
ax.set_xlabel("Participants",fontsize=12)
ax.set_ylabel("Scores",fontsize=12)
ax.set_xticklabels(x_labels)
grades.plot(kind='bar', title ="grades,scores-of-Participants",figsize=(15,10),legend=True,color = "red", fontsize=12)
ax.set_xlabel("Participants",fontsize=12)
ax.set_ylabel("grades",fontsize=12)
ax.set_xticklabels(x_labels)


datasetp1 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p1.csv")
datasetp2 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p2.csv")
datasetp3 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p3.csv")
datasetp4 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p4.csv")
datasetp5 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p5.csv")
datasetp6 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p6.csv")
datasetp7 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p7.csv")
datasetp8 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p8.csv")
datasetp9 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p9.csv")
datasetp10 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p10.csv")
datasetp11 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p11.csv")
datasetp12 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p12.csv")
datasetp13 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p13.csv")
datasetp14 = pd.read_csv(r"G:\Semester - 3\Project\Datasets\output\ir\p14.csv")

##dataframes = [datasetp1, datasetp2, datasetp3,datasetp4,datasetp5,datasetp6,
##          datasetp7, datasetp8, datasetp9, datasetp10, datasetp11,
##          datasetp12, datasetp13, datasetp14]

##count_nan = datasetp1.isnull().sum()
##count_without_nan = count_nan[count_nan == 0]
##clean_data = datasetp1[count_without_nan.keys()]
##print(clean_data)
#dataset = pd.concat(dataframes)
datasetp1.dropna(axis = 0, inplace = True)
datasetp2.dropna(axis = 0, inplace = True)
datasetp3.dropna(axis = 0, inplace = True)
datasetp4.dropna(axis = 0, inplace = True)
datasetp5.dropna(axis = 0, inplace = True)
datasetp6.dropna(axis = 0, inplace = True)
datasetp7.dropna(axis = 0, inplace = True)
datasetp8.dropna(axis = 0, inplace = True)
datasetp9.dropna(axis = 0, inplace = True)
datasetp10.dropna(axis = 0, inplace = True)
datasetp11.dropna(axis = 0, inplace = True)
datasetp12.dropna(axis = 0, inplace = True)
datasetp13.dropna(axis = 0, inplace = True)
datasetp14.dropna(axis = 0, inplace = True)
##check = np.asarray(np.where(dataset.isnull()))
##print(check.shape)

plt.subplot(121)
plt.title("Temperature-Trends")
plt.xlabel("Time")
plt.ylabel("Temperature")

plt.plot(datasetp1["time_from_start"], datasetp1["filtered_temperature"],color = "black", label="P1")
plt.plot(datasetp2["time_from_start"], datasetp2["filtered_temperature"],color = "yellow", label = "P2")
plt.plot(datasetp3["time_from_start"], datasetp3["filtered_temperature"],color = "orange", label="P3")
plt.plot(datasetp4["time_from_start"], datasetp4["filtered_temperature"], color = "cyan",label = "P4")
plt.plot(datasetp5["time_from_start"], datasetp5["filtered_temperature"],color = "green",label="P5")
plt.plot(datasetp6["time_from_start"], datasetp6["filtered_temperature"], color = "brown", label = "P6")
plt.plot(datasetp7["time_from_start"], datasetp7["filtered_temperature"],color = "red", label="P7")
plt.plot(datasetp8["time_from_start"], datasetp8["filtered_temperature"], color = "magenta", label = "P8")
plt.plot(datasetp9["time_from_start"], datasetp9["filtered_temperature"],color = "gray",label="P9")
plt.plot(datasetp10["time_from_start"], datasetp10["filtered_temperature"], color = "teal",label = "p10")
plt.plot(datasetp11["time_from_start"], datasetp11["filtered_temperature"],color = "pink",label="P11")
plt.plot(datasetp12["time_from_start"], datasetp12["filtered_temperature"],color = "blue", label = "p12")
plt.plot(datasetp13["time_from_start"], datasetp13["filtered_temperature"],color = "violet",label="P13")
plt.plot(datasetp14["time_from_start"], datasetp14["filtered_temperature"],color = "silver",label = "P14")
plt.legend()
plt.show()


