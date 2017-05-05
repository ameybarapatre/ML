
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import GridSearchCV






image_data = np.loadtxt('datait.out', delimiter=',', dtype='float32')
label_data = np.loadtxt('datalt.out', delimiter=',', dtype='float32')

print('Preprocessing.... ', len(image_data), len(label_data))


scaler = preprocessing.StandardScaler()
scaler.fit(image_data)
image_data = scaler.transform(image_data)
pca = PCA(n_components= 30 , whiten = True)
pca.fit(image_data)
print(pca.explained_variance_ratio_.sum())
image_data = pca.transform(image_data)
#image_data =  pca.inverse_transform(image_data)

"""
# Create a classifier: a support vector classifier

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(image_data, label_data)
print("Best estimator found by grid search:")
print(clf.best_estimator_)
# We learn the digits on the first half of the digits
"""
classifier =  svm.SVC()
classifier.fit(image_data , label_data)

image_data = np.loadtxt('datai.out', delimiter=',', dtype='float32')
image_data = scaler.transform(image_data)
image_data = pca.transform(image_data)
#image_data =  pca.inverse_transform(image_data)
label_data = np.loadtxt('datal.out', delimiter=',', dtype='float32')
# Now predict the value of the digit on the second half:
expected = label_data
predicted = classifier.predict(image_data)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
