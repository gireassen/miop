# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Загружаем данные ирисов Фишера
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

# Разделяем данные на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

# Создаем и обучаем классификатор k ближайших соседей
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# Предсказываем классы для тестовой выборки
y_pred = knn.predict(x_test)

# Оцениваем модель
accuracy = accuracy_score(y_test, y_pred)
print("Процент правильных результатов:", accuracy * 100)

misclassification = 1 - accuracy
print("Процент промахов:", misclassification * 100)

ari = adjusted_rand_score(y_test, y_pred)
print("Индекс скорректированной случайной согласованности:", ari)

f1 = f1_score(y_test, y_pred, average='macro')
print("F-мера:", f1)
