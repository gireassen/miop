# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def graph_2d():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    main_components = 3
    pca = PCA(n_components=main_components)
    x_pca = pca.fit_transform(x_scaled)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        ax.scatter(x_pca[y == i, 0], x_pca[y == i, 1], x_pca[y == i, 2], color=color, alpha=.8, lw=lw, label=target_name)

    ax.legend(loc='best', shadow=False, scatterpoints=1)
    ax.set_title('Метод главных компонент на данных ирисов Фишера')
    plt.show()

def graph_3d():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Стандартизируем данные
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Применяем метод главных компонент для уменьшения размерности до 2 компонент
    main_components = 2
    pca = PCA(n_components=main_components)
    x_pca = pca.fit_transform(x_scaled)

    # Визуализируем данные после PCA
    plt.figure(figsize=(8, 6))
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(x_pca[y == i, 0], x_pca[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Метод главных компонент на данных ирисов Фишера')
    plt.show()

def rate_the_model():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    main_components = 2
    pca = PCA(n_components=main_components)
    x_pca = pca.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("-"*50)
    print("Процент правильных результатов:", accuracy * 100)

    misclassification = 1 - accuracy
    print("Процент промахов:", misclassification * 100)

    ari = adjusted_rand_score(y_test, y_pred)
    print("Индекс скорректированной случайной согласованности:", ari)

    f1 = f1_score(y_test, y_pred, average='macro')
    print("F-мера:", f1)
    print("-"*30)

if __name__ == "__main__":
    graph_2d()
    graph_3d()
    rate_the_model()

