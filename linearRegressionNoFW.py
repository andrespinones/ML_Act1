#Andrés Piñones Besnier A01570150
#Regresion Linear Simple
#Implementacion de un modelo de machine learning sin framework 
#importamos las librerias necesarias
import pandas as pd
import numpy as np

def linearRegression(csv):
  # leemos el archivo de datos csv utilizando pandas dataframe
  columns = ["x","y"]
  df = pd.read_csv(csv,names = columns)

  train = df.iloc[:int(len(df)*0.70)]

  test = df.iloc[int(len(df)*0.70):]
  # declaramos nuestra variable independiente(x) y dependiente(y)
  # usaremos el 70% del dataframe para entrenar y evaluar el modelo y el otro 30 para pruebas y evaluar error medio
  train_X = train['x'].values
  train_Y = train['y'].values

  test_X = test['x'].values
  test_Y = test['y'].values

  #calculamos la media de x y y
  mean_x = np.mean(train_X)
  mean_y = np.mean(train_Y)

  # numero total de valores de input
  n = len(train_X)
  total = len(train_X)
  test = len(test_X)

  # usamos la formula para calcular m y b de y=mx+b
  numerator = 0
  denominator = 0
  for i in range(n):
    numerator += (mean_x - train_X[i]) * (mean_y - train_Y[i])
    denominator += (mean_x - train_X[i]) ** 2
  m = numerator / denominator
  b = mean_y - (m * mean_x)

  # calculamos R cuadrada para evaluar la precisión de nuestro modelo 

  ss_t = 0 #suma de cuadrados
  ss_r = 0 #suma de cuadrados de los residuos

  for i in range(total): 
    y_pred = m * train_X[i] + b # y = mx + b
    ss_t += (mean_y - train_Y[i]) ** 2
    ss_r += (y_pred - train_Y[i]) ** 2
  r2 = 1-ss_r/ss_t

  modelo = 'El modelo de regresión es: y = {m}x + {b} con una R cuadrada de: {r2}'.format(m=m, b=b,r2=r2)
  print(modelo)

  #probamos el modelo con el resto de datos para evaluar el error cuadrado medio 
  se = 0
  for i in range(test): 
    y_pred = m * test_X[i] + b # y = mx + b
    error = (test_Y[i] - y_pred)
    se += (error ** 2)
  mse = se/test

  error = 'La predicción con los valores de prueba tuvo MSE de: {mse}'.format(mse=mse)
  print(error)

#Dataset con dos variables dependiente e independiente (x,y)
# En el caso de este, encontré una relacion entre las columnas de petal length y petal width (unicas variables que dejé en este dataset)
# Se creó un modelo de regresion lineal para predecir el ancho del petalo basado en el largo
#funciona con más datasets de dos variables
linearRegression('iris.csv')











