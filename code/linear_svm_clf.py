import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#load dataset
df = pd.read_csv('data\\linear_separable_bin_clf.csv',
                 names=['label', 'features_1', 'features_2'])

# print(df)
# print(df.info())
# print(df.describe())


#plot the data
class_0 = df[df['label'] == -1]
class_1 = df[df['label'] == 1]
plt.scatter(
    class_0['features_1'],
    class_0['features_2'],
    edgecolors='black',
    marker='o',
    color='green',
    label='Class 0'
)
plt.scatter(
    class_1['features_1'],
    class_1['features_2'],
    edgecolors='black',
    marker='o',
    color='orange',
    label='Class 1'
)
plt.xlabel('Features 1')
plt.ylabel('Features 2')
plt.title('Linear Separable Dataset')
plt.legend()
plt.show()


#prepare dataset
data = df.to_numpy()
X, y = data[:, 1:], data[:, 0]
test_size = 0.3
random_state = 29
is_shuffle = True
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = test_size,
    random_state = random_state,
    shuffle = is_shuffle
)


#SVC model
classifier = SVC(kernel='linear', random_state=random_state)
classifier.fit(X_train, y_train)


#evaluate model
y_pred = classifier.predict(X_test)
print('Accuracy is: ', accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))


#plot the graph with SVC
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 0].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(class_0['features_1'], class_0['features_2'], s=20, edgecolor="black", marker='o', color='blue', label='Class 0')
plt.scatter(class_1['features_1'], class_1['features_2'], s=20, edgecolor="black", marker='o', color='green', label='Class 1')
plt.xlabel('Features 1')
plt.ylabel('Features 2')
plt.title('Linear Separable Dataset')
plt.legend()
plt.show()
