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

#cls = SVC(kernel = "linear")
cls = SVC(kernel = "rbf") 
#cls = KNeighborsClassifier()
#cls = SVR(kernel = "rbf")
classfier_scores = []

print "The process Started...."
t1 = time.time()                    

cls.fit(train_x1,train_Y1)
classfier_scores.append(cls.score(test_x1, ground_truth1))

cls.fit(train_x2,train_Y2)
classfier_scores.append(cls.score(test_x2, ground_truth2))

cls.fit(train_x3,train_Y3)
classfier_scores.append(cls.score(test_x3, ground_truth3))

cls.fit(train_x4,train_Y4)
classfier_scores.append(cls.score(test_x4, ground_truth4))

cls.fit(train_x5,train_Y5)
classfier_scores.append(cls.score(test_x5, ground_truth5))

cls.fit(train_x6,train_Y6)
classfier_scores.append(cls.score(test_x6, ground_truth6))

cls.fit(train_x7,train_Y7)
classfier_scores.append(cls.score(test_x7, ground_truth7))

cls.fit(train_x8,train_Y8)
classfier_scores.append(cls.score(test_x8, ground_truth8))

cls.fit(train_x9,train_Y9)
classfier_scores.append(cls.score(test_x9, ground_truth9))

cls.fit(train_x10,train_Y10)
classfier_scores.append(cls.score(test_x10, ground_truth10))

cls.fit(train_x11,train_Y11)
classfier_scores.append(cls.score(test_x11, ground_truth11))

cls.fit(train_x12, train_Y12)
classfier_scores.append(cls.score(test_x12, ground_truth12))

cls.fit(train_x13,train_Y13)
classfier_scores.append(cls.score(test_x13, ground_truth13))

cls.fit(train_x14,train_Y14)
classfier_scores.append(cls.score(test_x14, ground_truth14))
                

Y_pred1 = (cls.predict(test_x1))
Y_pred2 = (cls.predict(test_x2))
Y_pred3 = (cls.predict(test_x3))
Y_pred4 = (cls.predict(test_x4))
Y_pred5 = (cls.predict(test_x5))
Y_pred6 = (cls.predict(test_x6))
Y_pred7 = (cls.predict(test_x7))
Y_pred8 = (cls.predict(test_x8))
Y_pred9 = (cls.predict(test_x9))
Y_pred10 = (cls.predict(test_x10))
Y_pred11 = (cls.predict(test_x11))
Y_pred12 = (cls.predict(test_x12))
Y_pred13 = (cls.predict(test_x13))
Y_pred14 = (cls.predict(test_x14))

Y_pred = np.concatenate((Y_pred1,Y_pred2,Y_pred3,Y_pred4,Y_pred5,Y_pred6,Y_pred7,Y_pred8,
                         Y_pred9,Y_pred10,Y_pred11,Y_pred12,Y_pred13,Y_pred14))
Y_true = pd.concat([ground_truth1, ground_truth2,ground_truth3,ground_truth4,ground_truth5,ground_truth6,ground_truth7,
                    ground_truth8,ground_truth9,ground_truth10,ground_truth11,ground_truth12,ground_truth13,ground_truth14])

conf_arr = confusion_matrix(Y_pred, Y_true)
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



##x_train, x_test, Y_train, Y_test = train_test_split(train_x_transformed, train_Y, test_size=0.2, random_state=5)
##
##classifier1 = SVC(kernel = "linear", C = 2)
##classifier2 = SVC(kernel = "rbf", C = 2)
##
plt.show()

#print Y_pred1




