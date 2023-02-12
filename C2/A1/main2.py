
import sys
from PyQt5 import QtWidgets, uic
import numpy as np
import math as m
import matplotlib.pyplot as plt
import os
import decimal
import pandas as pd

class Perceptron ():

    def __init__(self,aprendizaje, umbral):
        self.aprendizaje = aprendizaje
        self.umbral = umbral
        self.pesos = []
        self.error = 0
        self.X = []
        self.Y = []
        self.ids = []


    def leer_Archivo(self,archivo):
        df = pd.read_csv(archivo)
        x=df.iloc[:,1:4].values
        y = df.iloc[:, 4].values
        ids = df.iloc[: , 0].values
        x = self.bias(x)
        self.ids = ids
        self.X = x
        self.Y = y
        self.pesos = np.random.rand(len(x[0]))
        print("Pesos iniciales:",self.pesos)

    def bias(self, x):
        x_bias = []
        for i in range(len(x)):
            x_bias.append([1,x[i][0], x[i][1],x[i][2]])
        #print("X_bias:",x_bias)
        return x_bias

    def calculo_u(self):
        transpuestaW = np.transpose(self.pesos)
        u = np.linalg.multi_dot([self.X, transpuestaW])
        return u

    def funcion_activacion(self, u):
        return u
    
    def cal_error(self,ycal):
        error = []
        for i in range(len(ycal)):
            error.append(self.Y[i]-ycal[i])
        return error

    def delta_W(self,e):
        ret = np.transpose(e)
        for i in range(len(self.pesos)):
            dw = np.linalg.multi_dot([ret,self.X])*self.aprendizaje
        return dw
    
    def nv_W(self,deltaW):
        nueva_W = self.pesos + deltaW
        self.pesos = nueva_W
        
        return nueva_W

    def cal_error2(self,error):

        #PARA EVITAR DESBORDE CON DECIMAL
        # e = decimal.Decimal(0)
        # n = len(error)
        # for i in range(len(error)):
        #     e = e + decimal.Decimal(error[i])**2
        # mse = e / n
        # rmse = m.sqrt(mse)
        # return rmse

        #PARA EVITAR EL DESBORDE DEL VALOR
        # e = 0
        # n = len(error)
        # error = np.clip(error, -1e10, 1e10)
        # for i in range(len(error)):
        #     e = e + error[i]**2
        # mse = e / n
        # rmse = m.sqrt(mse)
        # return rmse

        #LA COMPLETA SEGUN
        e = 0
        n = len(error)
        for i in range(len(error)):
            e = e + error[i]**2
        mse = e / n
        rmse = m.sqrt(mse)
        return rmse
        
        #LA NORMALITA
        # e = 0
        # for i in range(len(error)):
        #     e = e + error[i]**2
        # return m.sqrt(e)


    
    
def inicializacion_alg():
    bandera=True
    try:
        aprendizaje = float(window.lineEditTAprendizaje.text())
        umbral = float(window.lineEditEPermisible.text())
        if(aprendizaje > 1 or aprendizaje <= 0):
            window.labelMensaje.setText("El valor de aprendizaje debe estar entre 0 y 1")
            window.labelMensaje.setStyleSheet("color: red ; font-size: 10pt")
            bandera = False
        if(umbral <= 0):
            window.labelMensaje.setText("El valor del umbral debe ser mayor a 0")
            window.labelMensaje.setStyleSheet("color: red ; font-size: 10pt")
            bandera = False
        if(archivo == ""):
            window.labelMensaje.setText("Seleccione un archivo")
            window.labelMensaje.setStyleSheet("color: red ; font-size: 10pt")
            bandera = False
            
    except:
        bandera = False

    if bandera:
        nombre = archivo.split("/")
        nombre = nombre[len(nombre)-1].split(".")
        window.labelMensaje.setText("Generando Graficas "+str(nombre[0]))
        window.labelMensaje.setStyleSheet("color: Black ; font-size: 10pt")
        window.labelMensaje.repaint()
        algoritmo(aprendizaje, umbral,archivo)
    
def algoritmo(aprendizaje, umbral,archivo):
    perceptron = Perceptron(aprendizaje, umbral)
    a =str(aprendizaje)
    errores=[]
    pesosSesgo= []
    pesosX1 = []
    pesosX2 = []
    pesosX3 = []
    perceptron.leer_Archivo(archivo)
    e = 2000
    i = 0
    while e > perceptron.umbral:
        u = perceptron.calculo_u()
        ycal = perceptron.funcion_activacion(u)
        error = perceptron.cal_error(ycal)
        #print("Error antes de deltaW", error)
        graficar_error2(error, i)
        deltaW = perceptron.delta_W(error)
        pesosSesgo.append(perceptron.pesos[0])
        pesosX1.append(perceptron.pesos[1])
        pesosX2.append(perceptron.pesos[2])
        pesosX3.append(perceptron.pesos[3])
        perceptron.nv_W(deltaW)
        e = perceptron.cal_error2(error)

        print("e:",e)
        #print("deltaW",deltaW)
        #print("Pesos",perceptron.pesos)
        
        errores.append(e)
        i += 1     

    #print("errores:",errores)
    print("Pesos finales:", perceptron.pesos)
    print('Cantidad de epocas de entrenamiento:',i)
    print("Maximo error observado:",max(errores))
    # print('Vueltas t:',i)
    graficar_error(errores, a)
    graficar_yc(ycal, perceptron)
    pesos = [pesosSesgo,pesosX1,pesosX2,pesosX3]
    graficar_pesos(pesos)
    window.labelMensaje.setText("Graficas Generadas ")
    window.labelMensaje.setStyleSheet("color: Green ; font-size: 10pt")
    
    window.labelMensaje.update()
    
    
        
def graficar_error(errores, a):
    plt.title("Evolución de la magnitud del error")
    plt.xlabel("Iteracion")
    plt.ylabel("Valor del error")
    plt.plot(errores, label="TA:"+a,linestyle="-")
    plt.legend()
    os.makedirs("assets\Graficas\Error", exist_ok=True)
    plt.savefig("assets\Graficas\Error\Error.png")
    plt.close()

def graficar_error2(error,i):
    plt.title("Error observador:"+str(i))
    plt.xlabel("Identificador")
    plt.ylabel("Valor del error")
    plt.plot(error, label="Error Observado:",linestyle="-")
    plt.legend()
    os.makedirs("assets\Graficas\ErrorObservado", exist_ok=True)
    plt.savefig("assets\Graficas\ErrorObservado\ErrorObservado"+str(i)+".png")
    plt.close()

def graficar_pesos(pesos):
    plt.title("Evolución de los pesos")
    plt.xlabel("Iteracion")
    plt.ylabel("Valor")
    plt.plot(pesos[0], label="Sesgo", color="blue",linestyle="-")
    plt.plot(pesos[1], label="X1", color="green",linestyle="-")
    plt.plot(pesos[2], label="X2", color="red",linestyle="-")
    plt.plot(pesos[3], label="X3", color="black",linestyle="-")
    plt.legend()
    os.makedirs("assets\Graficas\Pesos", exist_ok=True)
    plt.savefig("assets\Graficas\Pesos\Pesos.png")
    plt.close()

def graficar_yc(ycal,perceptron):
    plt.xlabel("Identificador")
    plt.ylabel("Valores Yc y Yd")
    plt.plot(ycal, label="Yc", color="blue",linestyle="-")
    plt.plot(perceptron.Y, label="Yd", color="green",linestyle="--")
    plt.legend()
    os.makedirs("assets\Graficas\Ycalculada", exist_ok=True)
    plt.savefig("assets\Graficas\Ycalculada\Ycalculada.png")
    plt.close()
    
def abrirArchivo():
    global archivo
    
    archivo = ""
    archivo = QtWidgets.QFileDialog.getOpenFileName(None, 'Abrir Archivo', 'C:\\', 'Text Files (*.csv)')[0]
    window.labelMensaje.setText("Archivo Cargado")


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = uic.loadUi("main_window.ui")
    window.show()
    valueAprendizaje = "0.00000001"
    valueError = "1"
    window.lineEditTAprendizaje.setText(valueAprendizaje)
    window.lineEditEPermisible.setText(valueError)
    window.cargarArchivo.clicked.connect(abrirArchivo)
    window.pushButtonEjecutar.clicked.connect(inicializacion_alg)
    sys.exit(app.exec())
