import numpy as np
import pandas as pd
from sklearn.svm import SVC,SVR
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import pickle

dataset = pd.read_csv(r"G:\Semester - 3\Project\Datasets\raw\meta\conf_training_set.csv")
dataset = dataset.fillna(value = -9999)

df = pd.DataFrame(dataset)

df.sample(frac=1)
train_x = df.drop(['confidence', 'Participant'], axis = 1, inplace = False)
train_Y = dataset["confidence"]

train_x1 = df[df.Participant != 'P14']
train_Y1 = train_x1['confidence']
train_x1 = train_x1.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x1 = df[df.Participant == 'P14']
##scaler = preprocessing.StandardScaler().fit(train_x1)
##train_x1 = scaler.transform(train_x1)
ground_truth1 = test_x1['confidence']
test_x1 = test_x1.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x2 = df[df.Participant != 'P13']
train_Y2 = train_x2['confidence']
train_x2 = train_x2.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x2 = df[df.Participant == 'P13']
##scaler = preprocessing.StandardScaler().fit(train_x2)
##train_x2 = scaler.transform(train_x2)
ground_truth2 = test_x2['confidence']
test_x2 = test_x2.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x3 = df[df.Participant != 'P12']
train_Y3 = train_x3['confidence']
train_x3 = train_x3.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x3 = df[df.Participant == 'P12']
##scaler = preprocessing.StandardScaler().fit(train_x3)
##train_x3 = scaler.transform(train_x3)
ground_truth3 = test_x3['confidence']
test_x3 = test_x3.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x4 = df[df.Participant != 'P11']
train_Y4 = train_x4['confidence']
train_x4 = train_x4.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x4 = df[df.Participant == 'P11']
##scaler = preprocessing.StandardScaler().fit(train_x4)
##train_x4 = scaler.transform(train_x4)
ground_truth4 = test_x4['confidence']
test_x4 = test_x4.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x5 = df[df.Participant != 'P10']
train_Y5 = train_x5['confidence']
train_x5 = train_x5.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x5 = df[df.Participant == 'P10']
##scaler = preprocessing.StandardScaler().fit(train_x5)
##train_x5 = scaler.transform(train_x5)
ground_truth5 = test_x5['confidence']
test_x5 = test_x5.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x6 = df[df.Participant != 'P9']
train_Y6 = train_x6['confidence']
train_x6 = train_x6.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x6 = df[df.Participant == 'P9']
##scaler = preprocessing.StandardScaler().fit(train_x6)
##train_x6 = scaler.transform(train_x6)
ground_truth6 = test_x6['confidence']
test_x6 = test_x6.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x7 = df[df.Participant != 'P8']
train_Y7 = train_x7['confidence']
train_x7 = train_x7.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x7 = df[df.Participant == 'P8']
##scaler = preprocessing.StandardScaler().fit(train_x7)
##train_x7 = scaler.transform(train_x7)
ground_truth7 = test_x7['confidence']
test_x7 = test_x7.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x8 = df[df.Participant != 'P7']
train_Y8 = train_x8['confidence']
train_x8 = train_x8.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x8 = df[df.Participant == 'P7']
##scaler = preprocessing.StandardScaler().fit(train_x8)
##train_x8 = scaler.transform(train_x8)
ground_truth8 = test_x8['confidence']
test_x8 = test_x8.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x9 = df[df.Participant != 'P6']
train_Y9 = train_x9['confidence']
train_x9 = train_x9.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x9 = df[df.Participant == 'P6']
##scaler = preprocessing.StandardScaler().fit(train_x9)
##train_x9 = scaler.transform(train_x9)
ground_truth9 = test_x9['confidence']
test_x9 = test_x9.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x10 = df[df.Participant != 'P5']
train_Y10 = train_x10['confidence']
train_x10 = train_x10.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x10 = df[df.Participant == 'P5']
##scaler = preprocessing.StandardScaler().fit(train_x10)
##train_x10 = scaler.transform(train_x10)
ground_truth10 = test_x10['confidence']
test_x10 = test_x10.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x11 = df[df.Participant != 'P4']
train_Y11 = train_x11['confidence']
train_x11 = train_x11.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x11 = df[df.Participant == 'P4']
##scaler = preprocessing.StandardScaler().fit(train_x11)
##train_x11 = scaler.transform(train_x11)
ground_truth11 = test_x11['confidence']
test_x11 = test_x11.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x12 = df[df.Participant != 'P3']
train_Y12 = train_x12['confidence']
train_x12 = train_x12.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x12 = df[df.Participant == 'P3']
##scaler = preprocessing.StandardScaler().fit(train_x12)
##train_x12 = scaler.transform(train_x12)
ground_truth12 = test_x12['confidence']
test_x12 = test_x12.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x13 = df[df.Participant != 'P2']
train_Y13 = train_x13['confidence']
train_x13 = train_x13.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x13 = df[df.Participant == 'P2']
##scaler = preprocessing.StandardScaler().fit(train_x13)
##train_x13 = scaler.transform(train_x13)
ground_truth13 = test_x13['confidence']
test_x13 = test_x13.drop(['Participant', 'confidence'], axis = 1, inplace = False)
train_x14 = df[df.Participant != 'P1']
train_Y14 = train_x14['confidence']
train_x14 = train_x14.drop(['Participant', 'confidence'], axis = 1, inplace = False)
test_x14 = df[df.Participant == 'P1']
ground_truth14 = test_x14['confidence']
##scaler = preprocessing.StandardScaler().fit(train_x14)
##train_x14 = scaler.transform(train_x14)
test_x14 = test_x14.drop(['Participant', 'confidence'], axis = 1, inplace = False)

#cls = KNeighborsClassifier()
#cls = linear()
#cls = SVR(kernel = "linear")
classfier_scores = []

print "The process Started...."
t1 = time.time()                    

cls1 = KNeighborsClassifier() 
cls1.fit(train_x1,train_Y1)
classfier_scores.append(cls1.score(test_x1, ground_truth1))
Y_pred1 = (cls1.predict(test_x1))

cls2 = KNeighborsClassifier() 
cls2.fit(train_x2,train_Y2)
classfier_scores.append(cls2.score(test_x2, ground_truth2))
Y_pred2 = (cls2.predict(test_x2))

cls3 = KNeighborsClassifier() 
cls3.fit(train_x3,train_Y3)
classfier_scores.append(cls3.score(test_x3, ground_truth3))
Y_pred3 = (cls3.predict(test_x3))

cls4 = KNeighborsClassifier() 
cls4.fit(train_x4,train_Y4)
classfier_scores.append(cls4.score(test_x4, ground_truth4))
Y_pred4 = (cls4.predict(test_x4))

cls5 = KNeighborsClassifier() 
cls5.fit(train_x5,train_Y5)
classfier_scores.append(cls5.score(test_x5, ground_truth5))
Y_pred5 = (cls5.predict(test_x5))

cls6 = KNeighborsClassifier() 
cls6.fit(train_x6,train_Y6)
classfier_scores.append(cls6.score(test_x6, ground_truth6))
Y_pred6 = (cls6.predict(test_x6))

cls7 = KNeighborsClassifier() 
cls7.fit(train_x7,train_Y7)
classfier_scores.append(cls7.score(test_x7, ground_truth7))
Y_pred7 = (cls7.predict(test_x7))

cls8 = KNeighborsClassifier() 
cls8.fit(train_x8,train_Y8)
classfier_scores.append(cls8.score(test_x8, ground_truth8))
Y_pred8 = (cls8.predict(test_x8))

cls9 = KNeighborsClassifier() 
cls9.fit(train_x9,train_Y9)
classfier_scores.append(cls9.score(test_x9, ground_truth9))
Y_pred9 = (cls9.predict(test_x9))

cls10 = KNeighborsClassifier() 
cls10.fit(train_x10,train_Y10)
classfier_scores.append(cls10.score(test_x10, ground_truth10))
Y_pred10 = (cls10.predict(test_x10))

cls11 = KNeighborsClassifier() 
cls11.fit(train_x11,train_Y11)
classfier_scores.append(cls11.score(test_x11, ground_truth11))
Y_pred11 = (cls11.predict(test_x11))

cls12 = KNeighborsClassifier() 
cls12.fit(train_x12, train_Y12)
classfier_scores.append(cls12.score(test_x12, ground_truth12))
Y_pred12 = (cls12.predict(test_x12))

cls13 = KNeighborsClassifier() 
cls13.fit(train_x13,train_Y13)
classfier_scores.append(cls13.score(test_x13, ground_truth13))
Y_pred13 = (cls13.predict(test_x13))

cls14 = KNeighborsClassifier() 
cls14.fit(train_x14,train_Y14)
classfier_scores.append(cls14.score(test_x14, ground_truth14))
Y_pred14 = (cls14.predict(test_x14))                

Y_pred = np.concatenate((Y_pred1,Y_pred2,Y_pred3,Y_pred4,Y_pred5,Y_pred6,Y_pred7,Y_pred8,
                         Y_pred9,Y_pred10,Y_pred11,Y_pred12,Y_pred13,Y_pred14))
Y_true = pd.concat([ground_truth1, ground_truth2,ground_truth3,ground_truth4,ground_truth5,ground_truth6,ground_truth7,
                    ground_truth8,ground_truth9,ground_truth10,ground_truth11,ground_truth12,ground_truth13,ground_truth14])


conf_arr = confusion_matrix(Y_true, Y_pred)
print conf_arr
norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
         try:
              tmp_arr.append(float(j)/float(a))
         except Exception:
              pass
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
plt.title("Confusion Matrix For Answer Prediction")
ax = fig.add_subplot(111)
ax.set_aspect(1)

res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')


width, height = conf_arr.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
alphabet = ['0', '20', '40', '60', '80', '100']
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])


print "The process Ended"
print "Total time taken", time.time() - t1
print "Oveall_Classification_accuracy::", np.mean(classfier_scores)
print "\n"

plt.show()






