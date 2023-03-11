from shutil import rmtree
import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
import plotly.graph_objects as go
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from DNA import DNA

# Arreglo de habilidades
habilidad_valor = ['N/A',
    "Desarrollo de aplicaciones móviles",
    "Desarrollo de aplicaciones web",
    "Desarrollo de aplicaciones de escritorio",
    "Desarrollo de juegos",
    "Desarrollo de aplicaciones empresariales",
    "Desarrollo de software de inteligencia artificial",
    "Desarrollo de software de aprendizaje automático",
    "Desarrollo de software de realidad virtual y aumentada",
    "Desarrollo de software de seguridad informática",
    "Desarrollo de software de automatización de procesos empresariales",
    "Desarrollo de software de automatización de pruebas",
    "Desarrollo de software de análisis de datos",
    "Desarrollo de software de gestión de proyectos",
    "Desarrollo de software de gestión de contenido",
    "Desarrollo de software de gestión de relaciones con los clientes",
    "Desarrollo de software de gestión de recursos humanos",
    "Desarrollo de software de gestión financiera",
    "Desarrollo de software de gestión de inventarios",
    "Desarrollo de software de logística y transporte",
    "Desarrollo de software de simulación y modelado"
]


def generar_tabla(individuo):
    matriz_habilidades = []
    cant_habilidades_coincidentes = []
    for j in range(len(individuo[2])):
        habilidades_coincidentes = []
        for i in range(len(arreglo_habilidades)):
            if arreglo_habilidades[i] in individuo[5][j]:
                habilidades_coincidentes.append("Cumple")
            else:
                habilidades_coincidentes.append("No cumple")
        matriz_habilidades.append(habilidades_coincidentes)
    # print(matriz_habilidades)
    for i in range(len(matriz_habilidades[0])):
        cantidad_habilidades_coincidentes = 0
        for j in range(len(matriz_habilidades)):
            if matriz_habilidades[j][i] == "Cumple":
                cantidad_habilidades_coincidentes += 1
        cant_habilidades_coincidentes.append(cantidad_habilidades_coincidentes)
    fig = go.Figure(data=[go.Table(
    header=dict(values=['Habilidades'] + individuo[2] + ['Cantidad de habilidades coincidentes']),
    cells=dict(values=[arreglo_habilidades] + matriz_habilidades+[cant_habilidades_coincidentes])),
    ], layout=go.Layout(title='Tabla de mejor individuo')) 
    fig.show()

def guardar_individuo(individuo,i):
    habilidades_nombre = []
    for j in range(len(individuo[5])):
        habilidad_nombre = []
        for k in range(len(individuo[5][j])):
            habilidad_nombre.append(habilidad_valor[individuo[5][j][k]])
        habilidades_nombre.append(habilidad_nombre)
    fig = go.Figure(data=[go.Table(
        columnwidth = [30,30,130,30],
        header=dict(values=['Nombres de los integrantes ','Salario Individual','Rol que desempeña', 'Salario total al mes: '+str(individuo[3])]),
        cells=dict(values=[individuo[2],individuo[4],habilidades_nombre]))
        ])
    fig.write_image("salida\Imagenes\Tabla/Tabla"+str(i)+".png", width=1500, height=500)
    
def sin_ceros(individuo):
    matriz_habilidades = []
    cant_habilidades_coincidentes = []
    for j in range(len(individuo[2])):
        habilidades_coincidentes = []
        for i in range(len(arreglo_habilidades)):
            if arreglo_habilidades[i] in individuo[5][j]:
                habilidades_coincidentes.append(1)
            else:
                habilidades_coincidentes.append(0)
        matriz_habilidades.append(habilidades_coincidentes)
    for i in range(len(matriz_habilidades)):
        habilidades = matriz_habilidades[i]
        if sum(habilidades) == 0:
            return True
    return False
        
 
def main(dna):
    try:
        generaciones = []
        mejor_individuo = []
        promedio = []
        peor_individuo = []   
        
        csv = dna.abrir_csv(csv_file[0])  
        poblacion = dna.evaluar_poblacion(dna.generar_poblacion(csv))
        bandera = True
        contador = 0
        # interfaz.close()
        if dna.is_valid():
            while bandera:
                for i in range(50):
                        poblacionAntesPoda = dna.mutacion(dna.cruzar(dna.seleccion(poblacion), 0.95), 0.2, 0.2)
                        poblacionAntesPoda = dna.evaluar_poblacion(poblacionAntesPoda)
                        poblacionAntesPoda = dna.agregar_poblacion(poblacion, poblacionAntesPoda)
                        poblacionOrdenada = dna.ordenar_valores(poblacionAntesPoda)
                        mejor_individuo.append(poblacionOrdenada[0])
                        promedio.append(np.mean(poblacionOrdenada))
                        peor_individuo.append(poblacionOrdenada[-1])
                        poblacion = dna.poda(poblacionAntesPoda, 20)
                        generaciones.append(poblacion)
                        print("Generacion: " + str(i))
                bandera = sin_ceros(generaciones[-1][0])
                if bandera and contador < 10:
                    contador += 1
                elif contador == 10:
                    bandera = False
                    print("No se pudo encontrar una solucion en el tiempo esperado")
                    break
            try:
                rmtree("salida\Imagenes")
            except:
                pass 
            os.makedirs("salida\Imagenes\Tabla", exist_ok=True)
            for i in range(5):
                guardar_individuo(generaciones[-1][i],i)
                guardar_individuo(generaciones[-1].pop(),5+i)
            generar_tabla(generaciones[-1][0])
            plt.plot(mejor_individuo, label="Mejor individuo", color="red", linestyle="-",)
            plt.plot(promedio, label="Promedio", color="yellow", linestyle="-",)
            plt.plot(peor_individuo, label="Peor individuo", color="green", linestyle="-")
            plt.legend()
            os.makedirs("salida\Imagenes\GraficaHistorial/", exist_ok=True)
            plt.savefig("salida\Imagenes\GraficaHistorial/GraficaHistorial.png")
            os.makedirs("salida\Imagenes\Video", exist_ok=True)
            img = []   
            for i in range(10):
                img.append(cv2.imread("salida\Imagenes\Tabla/Tabla"+str(i)+".png"))
            alto, ancho = img[0].shape[:2]
            video = cv2.VideoWriter('salida\Imagenes\Video\mivideo.avi', cv2.VideoWriter_fourcc(*'DIVX'),3, (alto, ancho))
            for i in range(len(img)):
                video.write(img[i]) 
            print("OK")  
            interfaz.estado.setText("Se encontro una solucion")
            interfaz.estado.setStyleSheet("color: green")
        else:
            interfaz.estado.setText("No se pudo encontrar una solucion")
            interfaz.estado.setStyleSheet("color: yellow")    
    except Exception as e:
        print(e)
        interfaz.estado.setText("El no archivo no fue cargado o no se pudo abrir")
        interfaz.estado.setStyleSheet("color: red")
        print("El no archivo no fue cargado o no se pudo abrir")
        sys.exit(1)
        
        
arreglo_habilidades = []


def boton_cargar():
    global csv_file
    csv_file = QFileDialog.getOpenFileName(None, 'Abrir CSV de aspirantes',"","CSV(*.csv);;All Files(*.*)")
    interfaz.estado.update()
    interfaz.estado.setText("Archivo Cargado")
    interfaz.estado.setStyleSheet("color: Black ; font-size: 10pt")
    

def boton_iniciar(csv_file):
    arreglo_habilidades.clear()
    try:
        interfaz.estado.update()
        interfaz.estado.setText("Evaluando...")
        interfaz.estado.setStyleSheet("color: Blue ; font-size: 10pt")
        
        habilidades = interfaz.habilidades.text().split(",")
        for i in range(len(habilidades)):
            arreglo_habilidades.append(int(habilidades[i])) 
        
        
        dna = DNA(arreglo_habilidades) 
        main(dna)
        
    except Exception as e:
        interfaz.estado.setText("Error")
        interfaz.estado.setStyleSheet("color: Red ; font-size: 10pt")
        print("Error, valores no guardados")
        print(e)
        sys.exit(1)
    
            
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("interfaz.ui")
    interfaz.show()
    interfaz.cargarCSV.clicked.connect(boton_cargar)
    interfaz.iniciarPrograma.clicked.connect(boton_iniciar)
    sys.exit(app.exec())