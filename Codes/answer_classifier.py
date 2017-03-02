import numpy as np
import pandas as pd
from sklearn.svm import SVC,SVR
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import time
import pickle

dataset = pd.read_csv(r"G:\Semester - 3\Project\Datasets\raw\meta\training_set.csv")
dataset = dataset.fillna(value = -9999)

df = pd.DataFrame(dataset)
#df = shuffle(df)
df.sample(frac=1)

train_x = df.drop('Answers', axis = 1, inplace = False)
train_Y = dataset["Answers"]

train_x = np.asarray(train_x)
train_Y = np.asarray(train_Y)

print train_x

scaler = preprocessing.StandardScaler().fit(train_x)
train_x_transformed = scaler.transform(train_x)

x_train, x_test, Y_train, Y_test = train_test_split(train_x, train_Y, test_size=0.2, random_state=5)

classifier1 = SVC(kernel = "linear", C = 2)
classifier2 = SVC(kernel = "rbf", C = 2)

#classifier2 = SVR()

print "Training Started......."

time1 = time.time()
classifier1.fit(x_train,Y_train)

print "Training Ended"

print "Total time taken for training for Gaussian Kernel SVC:",time.time() - time1

print "Accuracy of the Gaussian Kernel SVC on test_data", classifier1.score(x_test, Y_test)

save_answer_classifier = open("answer_classifier_SVC_Linear.pickle", "wb")
pickle.dump(classifier1, save_answer_classifier)
save_answer_classifier.close()

print "Training Started......."

time2 = time.time()
classifier2.fit(x_train,Y_train)

print "Training Ended"

print "Total time taken for training for Linear Kernel SVC:",time.time() - time2

print "Accuracy of the Linear Kernel SVC on test_data", classifier2.score(x_test, Y_test)

save_answer_classifier = open("answer_classifier_SVC_Gaussian.pickle", "wb")
pickle.dump(classifier1, save_answer_classifier)
save_answer_classifier.close()

##just a sample test with correct value '0'

X = x_train[0].reshape(1,-1)
print classifier1.predict(X)
print classifier2.predict(X)

##print classifier2.predict(X)













