import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import warnings

warnings.filterwarnings("ignore")

data_path = (
    r"E:\DEPI\GIZ4_AIS2_S1\customer_purchases.csv"
)
data = pd.read_csv(data_path)

data.head()

x = data.iloc[:, [0, 1]].values
y = data.iloc[:, 2].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=2
)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = SVC(kernel="linear", random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(f1_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))

# importing the modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# plotting the fgiure
plt.figure(figsize=(7, 7))

# assigning the input values
X_set, y_set = x_train, y_train

# ploting the linear graph
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(("black", "white")),
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# ploting scattered graph for the values
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("red", "blue"))(i),
        label=j,
    )

# labeling the graph
plt.title("Purchased Vs Non-Purchased")
plt.xlabel("Salay")
plt.ylabel("Age")
plt.legend()
plt.show()

print("#" * 30)

cls_2 = SVC(kernel="rbf", random_state=0)
cls_2.fit(x_train, y_train)
y_pred_2 = cls_2.predict(x_test)

cm_2 = confusion_matrix(y_test, y_pred_2)
print(cm_2)
accuracy = accuracy_score(y_test, y_pred_2)
print("Accuracy:", accuracy)
print(f1_score(y_test, y_pred_2))
print(precision_score(y_test, y_pred_2))
print(recall_score(y_test, y_pred_2))

# plotting the fgiure
plt.figure(figsize=(7, 7))

# assigning the input values
X_set, y_set = x_test, y_test

# ploting the linear graph
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    cls_2.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(("black", "white")),
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# ploting scattered graph for the values
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("red", "blue"))(i),
        label=j,
    )

# labeling the graph
plt.title("Purchased Vs Non-Purchased")
plt.xlabel("Salay")
plt.ylabel("Age")
plt.legend()
plt.show()

print("#" * 30)

cm = confusion_matrix(y_test, y_pred, labels=cls_2.classes_)
sns.heatmap(cm, annot=True)
plt.savefig(
    r"E:\DEPI\GIZ4_AIS2_S1\visualizations"
)
plt.show()
