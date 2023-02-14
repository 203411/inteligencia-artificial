import sys
import matplotlib.pyplot as plt
import pandas as pd
import math as m
import tensorflow as tf
import numpy as np
from PyQt5 import QtWidgets, uic


class Neurona():

    def __init__(self, iteraciones, aprendizaje, umbral):
        self.iteraciones = iteraciones
        self.aprendizaje = aprendizaje
        self.umbral = umbral
        self.pesos = []
        self.error = 0
        self.X = []
        self.Y = []
        self.Xtensor = []
        self.Ytensor = []

    def leer_datos(self):
        df = pd.read_csv('datos2.csv')
        self.Xtensor = df.iloc[:, 0:3].values
        self.Ytensor = df.iloc[:, 3].values



    def leer_datos_alg(self):

        df = pd.read_csv('datos2.csv')
        x = df.iloc[:, 0:3].values
        y = df.iloc[:, 3].values 
        x = self.bias(x)
        self.X = x
        self.Y = y
        self.pesos = np.random.rand(len(x[0]))

    def bias(self, x):
        x_bias = []
        for i in range(len(x)):
            x_bias.append([1, x[i][0], x[i][1], x[i][2]])
        return x_bias
        
    def calcular_u(self):
        transpuestaW = np.transpose(self.pesos)
        u = np.linalg.multi_dot([self.X, transpuestaW])
        return u

    def funcion_activacion_lineal(self, u):
        return u

    def calcular_error(self, yc):
        error = []
        for i in range(len(yc)):
            error.append(self.Y[i] - yc[i])
        return error
    
    def delta_W(self, e):
        ret = np.transpose(e)
        for i in range(len(self.pesos)):
            dw = np.linalg.multi_dot([ret,self.X])*self.aprendizaje
        return dw

    def nueva_w(self, delta_w):
        nueva_W = self.pesos + deltaW
        self.pesos = nueva_W
        return nueva_W

    def calcular_e(self, error):
        #LA COMPLETA SEGUN
        e = 0
        n = len(error)
        for i in range(len(error)):
            e = e + error[i]**2
        mse = e / n
        rmse = m.sqrt(mse)
        return rmse
        

neurona = Neurona(100, 0.000001, 0.5)

def generar_grafica_aprendizajes_tensor():
    neurona = Neurona(100, 0.1, 0.5)
    neurona.leer_datos()
    error = []
    for i in range(3):
        capa = tf.keras.layers.Dense(units=1, input_shape=[3])
        modelo = tf.keras.Sequential([capa])
        modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.1 + i*0.1),
        loss='mean_squared_error')
        print("Comenzando entrenamiento...")
        historial = modelo.fit(neurona.Xtensor, neurona.Ytensor, epochs=100, verbose=False)
        print("Modelo entrenado!")
        error.append(historial.history["loss"])
        print("Pesos: ", modelo.get_weights())
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(error[0], label="Tensorflow", color="red", linewidth=2, linestyle="-")
    plt.plot(error[1], label="Tensorflow", color="green", linewidth=2, linestyle="-")
    plt.plot(error[2], label="Tensorflow", color="yellow", linewidth=2, linestyle="-")
    # plt.plot(error[3], label="Tensorflow", color="blue", linewidth=2, linestyle="-")
    # plt.plot(error[4], label="Tensorflow", color="purple", linewidth=2, linestyle="-")
    plt.show()


def generar_grafica_aprendizajes_alg():
    errores = []
    e = 100
    for i in range(3):
        e = []
        neurona = Neurona(100, 0.0000013 + i* 0.00000017, 0.5)
        neurona.leer_datos_alg()
        for j in range (neurona.iteraciones):
            u = neurona.calcular_u()
            yc = neurona.funcion_activacion_lineal(u)
            error = neurona.calcular_error(yc)
            delta_W = neurona.delta_W(error)
            neurona.nueva_w(delta_W)
            e.append(neurona.calcular_e(error))
        errores.append(e)
    plt.xlabel("# Iteracion")
    plt.ylabel("Magnitud de error")
    plt.plot(errores[0], label="Algoritmo", color="red", linewidth=2, linestyle="-")
    plt.plot(errores[1], label="Algoritmo", color="green", linewidth=2, linestyle="-")
    plt.plot(errores[2], label="Algoritmo", color="yellow", linewidth=2, linestyle="-")
    # plt.plot(errores[3], label="Algoritmo", color="blue", linewidth=2, linestyle="-")
    # plt.plot(errores[4], label="Algoritmo", color="purple", linewidth=2, linestyle="-")
    plt.show()

def generar_grafica_error_tensor(iteraciones, aprendizaje, umbral):
    neurona = Neurona(iteraciones, aprendizaje, umbral)
    neurona.leer_datos()
    capa = tf.keras.layers.Dense(units=1, input_shape=[3])
    modelo = tf.keras.Sequential([capa])
    modelo.compile(
    optimizer=tf.keras.optimizers.Adam(aprendizaje),
    loss='mean_squared_error')
    print("Comenzando entrenamiento...")
    historial = modelo.fit(neurona.Xtensor, neurona.Ytensor, epochs=iteraciones, verbose=False)
    print("Modelo entrenado!")
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(historial.history["loss"])
    scores = modelo.evaluate(neurona.Xtensor, neurona.Ytensor)
    print("Error: ", scores)
    print("Pesos: ", modelo.get_weights())
    plt.show()

def generar_grafica_error_alg(iteracion, aprendizaje, umbral):
    neurona = Neurona(iteracion, aprendizaje, umbral)
    errores = []
    neurona.leer_datos_alg()
    i = 0
    e = 100
    while neurona.iteraciones > i and e > neurona.umbral:
        u = neurona.calcular_u()
        yc = neurona.funcion_activacion_lineal(u)
        error = neurona.calcular_error(yc)
        delta_W = neurona.delta_W(error)
        neurona.nueva_w(delta_W)
        e = neurona.calcular_e(error)
        errores.append(e)
        print(e)
        i += 1

    print("Pesos: ", neurona.pesos)
    plt.plot(errores)
    plt.show()

def generar_grafica_yc():
    yct = tensor()
    yca = alg()
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(yct, label="Tensorflow", color="red", linewidth=2, linestyle="-")
    plt.plot(yca, label="Algoritmo", color="blue", linewidth=3,linestyle="-.")
    plt.plot(neurona.Y, label="Datos", color="green",linewidth=4 ,linestyle="--")
    plt.legend()
    plt.show()

def tensor():
    yc = []
    neurona_Tensor = Neurona(100, 0.0000013, 0.5)
    neurona_Tensor.leer_datos()
    capa = tf.keras.layers.Dense(units=1, input_shape=[3])
    modelo = tf.keras.Sequential([capa])
    modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error')
    print("Comenzando entrenamiento...")
    historial = modelo.fit(neurona_Tensor.Xtensor, neurona_Tensor.Ytensor, epochs=neurona.iteraciones, verbose=False)
    print("Modelo entrenado!")
    scores = modelo.evaluate(neurona_Tensor.Xtensor, neurona_Tensor.Ytensor)
    yc.append(modelo.predict(neurona_Tensor.Xtensor).flatten())
    print(modelo.get_weights())
    
    return yc[0]
    

def alg():
    errores = []
    neurona.leer_datos_alg()
    i = 0
    e = 100
    while neurona.iteraciones > i and e > neurona.umbral:
        u = neurona.calcular_u()
        yc = neurona.funcion_activacion_lineal(u)
        error = neurona.calcular_error(yc)
        delta_W = neurona.delta_W(error)
        neurona.nueva_w(delta_W)
        e = neurona.calcular_e(error)
        errores.append(e)
        i += 1
    return (yc)

def iniciar_valores_alg():

    bandera = True
    try:
        aprendizaje = float(interfaz.aprendizaje.text())
        iteraciones = int(interfaz.iteraciones.text())
        umbral = float(interfaz.error_permisible.text())
    except:
        interfaz.estado.setText("Error en los valores")
        bandera = False
    
    if bandera:
        interfaz.estado.setText("")
        generar_grafica_error_alg(iteraciones, aprendizaje, umbral)

def iniciar_valores_tensor():

    bandera = True

    try:
        aprendizaje = float(interfaz.aprendizaje.text())
        iteraciones = int(interfaz.iteraciones.text())
        umbral = float(interfaz.error_permisible.text())
    except:
        interfaz.estado.setText("Error en los valores")
        bandera = False
    
    if bandera:
        interfaz.estado.setText("")
        generar_grafica_error_tensor(iteraciones, aprendizaje, umbral)



if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("Main.ui")
    interfaz.show()
    interfaz.grafica_errores.clicked.connect(iniciar_valores_alg)
    interfaz.grafica_errores_2.clicked.connect(iniciar_valores_tensor)
    interfaz.grafica_comparativa.clicked.connect(generar_grafica_yc)
    interfaz.generar_grafica_aprendizajes.clicked.connect(generar_grafica_aprendizajes_alg)
    interfaz.generar_grafica_aprendizajes_2.clicked.connect(generar_grafica_aprendizajes_tensor)
    sys.exit(app.exec())