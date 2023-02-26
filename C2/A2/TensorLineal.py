import sys
import os
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

        df = pd.read_csv('203411.csv')
        self.Xtensor = df.iloc[:, 0:3].values
        self.Ytensor = df.iloc[:, 3].values


def iniciar_tensor(aprendizaje, umbral, epocas):
    neurona = Neurona(aprendizaje, umbral, epocas)
    neurona.leer_datos()

    yc = []
    pesos = []
    
    capa = tf.keras.layers.Dense(units=1, input_shape=[3])
    modelo = tf.keras.Sequential([capa])
    modelo.compile(
    optimizer=tf.keras.optimizers.Adam(aprendizaje),
    loss='mean_squared_error')
    print("Comenzando entrenamiento...")
    historial = modelo.fit(neurona.Xtensor, neurona.Ytensor, epochs=epocas, verbose=False)
    print("Modelo entrenado!")
    scores = modelo.evaluate(neurona.Xtensor, neurona.Ytensor)
    yc.append(modelo.predict(neurona.Xtensor).flatten())
    pesos.append(modelo.get_weights())

    print("Error: ", historial.history["loss"])
    print("Errores2:",scores)
    print("Yc: ", yc[0])
    #print("Pesos: ", pesos)

    grafica_error(historial, aprendizaje)
    graficar_Yc(yc,neurona)
    interfaz.labelMensaje.setText("Graficas Generadas ")
    interfaz.labelMensaje.setStyleSheet("color: Green ; font-size: 10pt")
    #graficar_pesos(pesos)

def grafica_error(historial,aprendizaje): 
    plt.title("Evolución de la magnitud del error")
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(historial.history["loss"],label="TA"+str(aprendizaje), linestyle="-")
    plt.legend()
    os.makedirs("assets\Graficas\Error", exist_ok=True)
    plt.savefig("assets\Graficas\Error\Error.png")
    plt.close()
    # plt.plot(error, label="Tensorflow", color="red", linewidth=2, linestyle="-")

def graficar_Yc(yc,neurona):
    plt.xlabel("Identificador")
    plt.ylabel("Valores Yc y Yd")
    plt.plot(yc[0], label="Yc", color="red", linewidth=1, linestyle="-")
    plt.plot(neurona.Ytensor, label="Yd", color="green",linewidth=2 ,linestyle="--")
    plt.legend()
    os.makedirs("assets\Graficas\Ycalculada", exist_ok=True)
    plt.savefig("assets\Graficas\Ycalculada\Ycalculada.png")
    plt.close()

def graficar_pesos():
    pass

def iniciar_valores_tensor():

    bandera = True

    try:

        aprendizaje = float(interfaz.lineEditTAprendizaje.text())
        umbral = float(interfaz.lineEditEPermisible.text())
        epocas = int(interfaz.lineEditEpocas.text())

    except:
        interfaz.labelMensaje.setText("Error en los valores")
        interfaz.labelMensaje.setStyleSheet("color: Red ; font-size: 10pt")
        interfaz.labelMensaje.repaint()
        bandera = False
    
    if bandera:
        #interfaz.estado.setText("")
        interfaz.labelMensaje.setText("Generando Graficas ")
        interfaz.labelMensaje.setStyleSheet("color: Black ; font-size: 10pt")
        interfaz.labelMensaje.repaint()
        iniciar_tensor(aprendizaje, umbral, epocas)
        #generar_grafica_error_tensor(iteraciones, aprendizaje, umbral)



if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("main_window.ui")
    interfaz.show()
    valueAprendizaje = "0.5"
    valueError = "0.01"
    valueEpocas = "100"

    interfaz.lineEditTAprendizaje.setText(valueAprendizaje)
    interfaz.lineEditEPermisible.setText(valueError)
    interfaz.lineEditEpocas.setText(valueEpocas)
    interfaz.pushButtonEjecutar.clicked.connect(iniciar_valores_tensor)
    sys.exit(app.exec())