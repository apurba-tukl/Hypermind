import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

cd = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(cd + "/../Datafiles/conf_training_set.csv")
df = df.fillna(value=-9999)

labels = []
predicteds = []
classfier_scores = []
for p in range(1, 15):
    training = df[df.Participant != "P"+str(p)]
    X_train = training.drop(["Participant", "confidence"], axis=1, inplace=False)
    y_train = training["confidence"]

    testing = df[df.Participant == "P"+str(p)]
    X_test = testing.drop(["Participant", "confidence"], axis=1, inplace=False)
    y_test = testing["confidence"]
    labels.extend(y_test)

    clf = SVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    predicteds.extend(predicted)
    classfier_scores.append(clf.score(X_test, y_test))

conf_arr = confusion_matrix(labels, predicteds)
print conf_arr

size = len(conf_arr)
norm_conf = [[0 for x in range(size)] for x in range(size)]
for j in range(size):
    s = np.sum(conf_arr[j])
    for i in range(size):
        norm_conf[j][i] = float(conf_arr[j][i])/float(s)

fig = plt.figure()
plt.clf()
plt.title("Confusion Matrix for Confidence Prediction")
ax = fig.add_subplot(111)
ax.set_aspect(1)

res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                interpolation='nearest')

for x in xrange(size):
    for y in xrange(size):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
alphabet = [str(x) for x in range(0, 100, 20)]
plt.xticks(range(size), alphabet[:size])
plt.yticks(range(size), alphabet[:size])
plt.ylabel('True label')
plt.xlabel('Predicted label')

print "The process Ended"
print "Oveall_Classification_accuracy::", np.mean(classfier_scores)
print "\n"

plt.show()
