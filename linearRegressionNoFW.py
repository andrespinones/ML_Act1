#Andrés Piñones Besnier A01570150
#Regresion Linear Simple
#Implementacion de un modelo de machine learning sin framework 
#importamos las librerias necesarias
import pandas as pd
import numpy as np

def create_df(csv):
  # leemos el archivo de datos csv utilizando pandas dataframe
  columns = ["x","y"]
  df = pd.read_csv(csv,names = columns)
  return df
  
def getXY(df):
  # declaramos nuestra variable independiente(x) y dependiente(y)
  X = df['x'].values
  Y = df['y'].values
  return(X,Y)

def getCoefficients(X,Y):
  #calculamos la media de x y y
  mean_x = np.mean(X)
  mean_y = np.mean(Y)

  # numero total de valores de input
  n = len(X)

  # usamos la formula para calcular m y b de y=mx+b
  numerator = 0
  denominator = 0
  for i in range(n):
    numerator += (mean_x - X[i]) * (mean_y - Y[i])
    denominator += (mean_x - X[i]) ** 2
  m = numerator / denominator
  b = mean_y - (m * mean_x)
  return(m,b)

  # calculamos R cuadrada para evaluar la precisión de nuestro modelo 
def r_cuadrada(n,m,b,X,Y,mean_y):
  ss_t = 0 #suma de cuadrados
  ss_r = 0 #suma de cuadrados de los residuos

  for i in range(n): 
    y_pred = m * X[i] + b # y = mx + b
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred ) ** 2
  r2 = 1-ss_r/ss_t

  modelo = 'El modelo de regresión es: y = {m}x + {b} con una R cuadrada de: {r2}'.format(m=m, b=b,r2=r2)
  print(modelo)

def mse(n,m,b,X,Y):
  #calculamos el MSE de la predicción para evaluar el error
  se = 0
  for i in range(n): 
    y_pred = m * X[i] + b # y = mx + b
    error = (Y[i] - y_pred)
    se += (error ** 2)
  mse = se/n

  error = 'La predicción con los valores tuvo MSE de: {mse}'.format(mse=mse)
  print(error)

def scratch_linearRegression(csv):
  df = create_df(csv)
  X,Y = getXY(df)
  m,b = getCoefficients(X,Y)
  r_cuadrada(len(X),m,b,X,Y,np.mean(Y))
  mse(len(X),m,b,X,Y)


#Dataset con dos variables dependiente e independiente (x,y)
# En el caso de este, encontré una relacion entre las columnas de petal length y petal width (unicas variables que dejé en este dataset)
# Se creó un modelo de regresion lineal para predecir el ancho del petalo basado en el largo
# funciona con datasets de dos variables
scratch_linearRegression('iris.csv')











