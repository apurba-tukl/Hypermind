import numpy as np
import itertools
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

cd = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(r"G:\Semester - 3\Project\Datasets\raw\meta\conf_training_set.csv")
df = dataset.fillna(value=-9999)

labels = []
predicteds = []
classfier_scores = []
class_names = ["High", "low"]


#df.sample(frac=1)
#df = shuffle(df, random_state=3)
train_x = df.drop(['confidence', 'Participant'], axis = 1, inplace = False)
train_Y = dataset["confidence"]
scaler = preprocessing.StandardScaler().fit(train_x)
train_x_transformed = scaler.transform(train_x)

for i in range(100):
    x_train, x_test, Y_train, Y_test = train_test_split(train_x_transformed, train_Y, test_size=0.1)
    clf = SVC(kernel = "linear", class_weight="balanced", C =.09)
    clf.fit(x_train, Y_train)
    #print clf.score(x_test, Y_test)
    y_pred = clf.predict(x_test)
    predicteds.extend(y_pred)
    labels.extend(Y_test)
    classfier_scores.append(clf.score(x_test, Y_test))
print f1_score(Y_test, y_pred, average=None)
print np.mean(classfier_scores)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(labels, predicteds)
print cnf_matrix
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()
