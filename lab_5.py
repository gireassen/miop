import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv('carsmall.csv')
'''
mpg: Расход топлива (miles per gallon)
cyl: Количество цилиндров
disp: Объём двигателя (displacement in cubic inches)
hp: Лошадиные силы
drat: Удельные обороты (axle ratio)
wt: Вес (thousands lbs)
qsec: Время заезда на 1/4 мили
vs: Двигатель V/S
am: Тип трансмиссии (0 = автоматическая, 1 = ручная)
gear: Количество передач
carb: Количество карбюраторов
'''

X = data[["hp"]]   # независимая переменная [для группы данных data[["hp", "am"]]]
y = data["mpg"]  #зависимая переменная

X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

#предсказания и доверительные интервалы
predictions = results.predict(X)
predictions_interval = results.get_prediction(X).conf_int(alpha=0.05)  # 95% доверительные интервалы

const_is = results.params['const']
hp_is = results.params['hp']
print(results.summary())

#график
plt.scatter(X["hp"], y, color='blue', label='наблюдения')
plt.plot(X["hp"], predictions, color='red', linewidth=2, 
         label=f'линия тренда y={round(const_is, 2)} + {round(hp_is, 2)}*x')
plt.fill_between(X["hp"], predictions_interval[:,0], predictions_interval[:,1], color='pink', 
                 alpha=.2, label='95% доверительный интервал')
plt.xlabel('hp: Лошадиные силы')
plt.ylabel('mpg: Расход топлива (miles per gallon)')
plt.legend()
plt.show()