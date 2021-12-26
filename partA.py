# Importing python packages
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import time  # to measure time taken by a model

dataset = pd.read_csv('heart.csv')

# refactoring data fields for better processing
temp_replace = {'M': 1, 'F': 0}
dataset.Sex = [temp_replace[temp_replace_loop] for temp_replace_loop in dataset.Sex]
# or dataset['Sex'].replace({'M':1,'F':0}, inplace = True)

temp_replace = {'TA': 0, 'ASY': 1, 'ATA': 2, 'NAP': 3}
dataset.ChestPainType = [temp_replace[temp_replace_loop] for temp_replace_loop in dataset.ChestPainType]

temp_replace = {'Down': 0, 'Flat': 1, 'Up': 2}
dataset.ST_Slope = [temp_replace[temp_replace_loop] for temp_replace_loop in dataset.ST_Slope]

temp_replace = {'Normal': 0, 'LVH': 1, 'ST': 2}
dataset.RestingECG = [temp_replace[temp_replace_loop] for temp_replace_loop in dataset.RestingECG]

temp_replace = {'N': 1, 'Y': 0}
dataset.ExerciseAngina = [temp_replace[temp_replace_loop] for temp_replace_loop in dataset.ExerciseAngina]

dataset.head()
x = dataset.drop('HeartDisease', axis=1)
y = dataset['HeartDisease']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=21)
# test_size = 0.2 as per given requirment in question paper
# SVM Training and Testing
SVM_Model = svm.SVC(kernel='linear')
start = time.time()
SVM_Model.fit(train_x, train_y)
end = time.time()
total_time_taken = end - start
start = time.time()
Prediction_of_SVM = SVM_Model.predict(test_x)
end = time.time()
testingTimeSVM = end - start
print('Efficiency of SVM model: ')
print('Training Time: ', "{:.4f}".format(total_time_taken))
print('Testing Time: ', "{:.4f}".format(testingTimeSVM))
Accuracy = SVM_Model.score(test_x, test_y) * 100
print('Accuracy: ', "{:.2f}".format(Accuracy), '%')
confusion_matrix(test_y, Prediction_of_SVM)
error = []
for k in range(1, 30):
    knn = KNeighborsClassifier(k)
    knn.fit(train_x, train_y)
    _predict = knn.predict(test_x)
    error.append(np.mean(_predict != test_y))

print("Minimum error: ", "{:.4f}".format(min(error)), "at K =", error.index(min(error)) + 1)
print("------------------------------------------")
# KNN Training and Testing
KNN_Model = KNeighborsClassifier(n_neighbors=5)
start = time.time()
KNN_Model.fit(train_x, train_y)
end = time.time()
triningTimeKNN = end - start
start = time.time()
Prediction_KNN = KNN_Model.predict(test_x)
end = time.time()
testingTimeKNN = end - start
print('Efficiency of KNN model: ')
print('Training Time: ', "{:.4f}".format(triningTimeKNN))
print('Testing Time: ', "{:.4f}".format(testingTimeKNN))
Accuracy = KNN_Model.score(test_x, test_y) * 100
print('Accuracy: ', "{:.2f}".format(Accuracy), '%')
print("------------------------------------------")
confusion_matrix(test_y, Prediction_KNN)
# Decision Tree Training and Testing
tree = DecisionTreeClassifier(max_depth=3)
start = time.time()
tree.fit(train_x, train_y)
end = time.time()
triningTimeTree = end - start
start = time.time()
Prediction_Tree = tree.predict(test_x)
end = time.time()
testingTimeTree = end - start
print('Efficiency of Tree model: ')
print('Training Time: ', "{:.4f}".format(triningTimeTree))
print('Testing Time: ', "{:.4f}".format(testingTimeTree))
Accuracy = tree.score(test_x, test_y) * 100
print('Accuracy: ', "{:.2f}".format(Accuracy), '%')
print("------------------------------------------")
confusion_matrix(test_y, Prediction_Tree)
_MLPClassifier = MLPClassifier(random_state=1)
MLPClassifier_Tuned = MLPClassifier(activation='tanh', solver='sgd', alpha=0.05, hidden_layer_sizes=(4000, 10),
                                    random_state=1)
start = time.time()
_MLPClassifier.fit(train_x, train_y)
end = time.time()
triningTimeTree = end - start
start = time.time()
Prediction_MLPClassifier = _MLPClassifier.predict(test_x)
end = time.time()
testingTimeTree = end - start
print('Efficiency of Neural model: ')
print('Training Time: ', "{:.4f}".format(triningTimeTree))
print('Testing Time: ', "{:.4f}".format(testingTimeTree))
Accuracy = tree.score(test_x, test_y) * 100
print('Accuracy: ', "{:.2f}".format(Accuracy), '%')
confusion_matrix(test_y, Prediction_MLPClassifier)
