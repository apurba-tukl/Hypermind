import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import style
from pandas import DataFrame

style.use("fivethirtyeight")
#fig = plt.figure()
dataset = pd.read_csv(r"G:\Semester - 3\Project\Datasets\raw\meta\questionnaire.csv")
#clean_dataset = dataset.dropna()
clean_dataset = dataset.fillna(value = 0)

importance = []

importance.append(scipy.stats.pearsonr(clean_dataset['Question1'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question2'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question3'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question4'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question5'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question6'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question7'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question8'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question9'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question10'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question11'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question12'], clean_dataset['Grade']))
importance.append(scipy.stats.pearsonr(clean_dataset['Question13'], clean_dataset['Grade']))

correlations = []

for i in range(len(importance)):
     correlations.append(importance[i][0])
correlations = pd.DataFrame(correlations)
p_values = []

for i in range(len(importance)):
     p_values.append(importance[i][1])
print correlations
print p_values
p_values = pd.DataFrame(p_values)
x_labels = ["QS-1", "QS-2","QS-3", "QS-4","QS-5", "QS-6","QS-7", "QS-8","QS-9", "QS-10","QS-11", "QS-12","QS-13"]

#ax1 = fig.add_subplot(121)
ax1 = correlations.plot(kind='bar', title ="Importance-of-questionaire",figsize=(15,10),legend=False,color = "blue", fontsize=12)
ax1.set_xlabel("Questions",fontsize=12)
ax1.set_ylabel("Correlation_Score",fontsize=12)
#ax1.set_ylim(-1,1,.2)
ax1.set_yticks([-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
ax1.set_xticklabels(x_labels)
rects = ax1.patches
labels = ["label%d" % i for i in xrange(len(rects))]
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

#ax2 = fig.add_subplot(122)
ax2 = p_values.plot(kind='bar', title ="P-Values",figsize=(15,10),legend=False,color = "brown", fontsize=12)
ax2.set_xlabel("Questions",fontsize=12)
ax2.set_ylabel("P-Value",fontsize=12)
#ax2.set_ylim(0,1,.2)
ax2.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
ax2.set_xticklabels(x_labels)
rects = ax2.patches
labels = ["label%d" % i for i in xrange(len(rects))]
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()
plt.savefig("D:\image.png")

