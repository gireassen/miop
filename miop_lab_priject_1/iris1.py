# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets # pip install scikit-learn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from scipy.stats import mode

# Загрузим набор данных
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head()


x = iris_df.iloc[:, [2, 3]].values


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

def one_and_two_row():
    '''
    первый и второй столбец
    '''
    plt.scatter(x[y_kmeans == 1, 1], 
                x[y_kmeans == 1, 2], 
                s = 100, 
                c = 'red', 
                label = 'Ирис щетинистый')

    plt.scatter(x[y_kmeans == 1, 0], 
                x[y_kmeans == 1, 1], 
                s = 100, 
                c = 'blue', 
                label = 'Ирис разноцветный')

    plt.scatter(x[y_kmeans == 2, 0], 
                x[y_kmeans == 2, 1], 
                s = 100, 
                c = 'green', 
                label = 'Ирис виргинский')

    plt.scatter(kmeans.cluster_centers_[:, 0], 
                kmeans.cluster_centers_[:,1], 
                s = 100, 
                c = 'yellow', 
                label = 'Центроиды')
    return plt

def two_and_three_row():
    '''
    второй и трейтий столбец
    '''
    plt.scatter(x[y_kmeans == 0, 0],
            x[y_kmeans == 0, 1],
            s = 100,
            c = 'red',
            label = 'Ирис щетинистый')

    plt.scatter(x[y_kmeans == 1, 0],
                x[y_kmeans == 1, 1],
                s = 100,
                c = 'blue',
                label = 'Ирис разноцветный')

    plt.scatter(x[y_kmeans == 2, 0],
                x[y_kmeans == 2, 1],
                s = 100,
                c = 'green',
                label = 'Ирис виргинский')

    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:,1],
                s = 100,
                c = 'yellow',
                label = 'Центроиды')
    return plt

def three_and_four_row():
    '''
    трейтий и 4й столбец
    '''
    plt.scatter(x[y_kmeans == 0, 2], 
            x[y_kmeans == 0, 3], 
            s = 100, 
            c = 'red', 
            label = 'Ирис щетинистый')

    plt.scatter(x[y_kmeans == 1, 2], 
                x[y_kmeans == 1, 3], 
                s = 100, 
                c = 'blue', 
                label = 'Ирис разноцветный')

    plt.scatter(x[y_kmeans == 2, 2], 
                x[y_kmeans == 2, 3], 
                s = 100, 
                c = 'green', 
                label = 'Ирис виргинский')

    plt.scatter(kmeans.cluster_centers_[:, 2], 
                kmeans.cluster_centers_[:, 3], 
                s = 100, 
                c = 'yellow', 
                label = 'Центроиды')
    return plt

res = two_and_three_row()
res.legend()
res.show()

#------------------------------------------------------------
y_true = iris.target
cluster_to_label_mapping = {}
for i in np.unique(y_kmeans):
    cluster_to_label_mapping[i] = mode(y_true[y_kmeans == i])[0]
    
y_kmeans_converted = [cluster_to_label_mapping[i] for i in y_kmeans]

kmeans.fit(x)
y_pred = kmeans.labels_

misclassification_rate = 1 - accuracy_score(y_true, y_kmeans_converted)
accuracy = accuracy_score(y_true, y_kmeans_converted) * 100
rand_index = adjusted_rand_score(y_true, y_kmeans_converted)
f_measure = f1_score(y_true, y_kmeans_converted, average='weighted')

print("Процент промахов:", misclassification_rate * 100)
print("Процент правильных результатов:", accuracy)
print("Индекс сходства:", rand_index)
print("F-мера:", f_measure)
#------------------------------------------------------------