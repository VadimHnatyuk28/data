import pandas as pd
import numpy as np

def main():
    x1 = np.array(pd.read_csv('data.csv')['income'].values)
    x2 = np.array(pd.read_csv('data.csv')['coefficient'].values)
    y = np.array(pd.read_csv('data.csv')['profitability'].values)

    # Додамо стовпець одиниць та фактори до матриці X для зручності розрахунків
    X = np.vstack([x1, x2, np.ones(len(x1))]).T

    # Застосуємо метод найменших квадратів для знаходження параметрів
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

    # Розділимо коефіцієнти на окремі змінні
    c, b, a = coefficients

    # Виведення результатів
    print("Коефіцієнт a рівний:", a)
    print("Коефіцієнт b рівний:", c)
    print("Коефіцієнт c рівний:", b)
    print(f"Рівняння регресії має наступний вигляд:Y = {round(a,3)} + {round(c,3)} * X1 + {round(b,3)} * X2 ")

    print("Для проведення прогнозованого розрахунку, задайте велечини чистого доходу та коефіцієнту фінансової незалежності")

    income=float(input("Введіть чистий дохід "))
    koef=float(input("Введіть коефіцієнт фінансової незалежності "))

    y = a+c*income+b*koef

    print(f"При величині чистого прибутку в {income} та величині коефіцієнті фінансової незалежності в {koef} , рентабельність підриємства рівне {y}")

if __name__ == '__main__':
   print(main())
