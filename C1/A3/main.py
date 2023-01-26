import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from PyQt5 import QtWidgets, uic
from shutil import rmtree
import plotly.graph_objects as go
import os
import cv2
from random import normalvariate, uniform, sample

class DNA():
    pasajeros_id =[]
    
    def __init__(self, poblacion_i, poblacion_f, pmi, pmg, p_cruza, generaciones, filas,ancho_pasillo, maximizar = True, verbose = True):
        self.poblacion_i = poblacion_i
        self.poblacion_f = poblacion_f
        self.pmi = pmi
        self.pmg = pmg
        self.p_cruza = p_cruza
        self.generaciones = generaciones
        self.filas = filas
        self.cantidad = filas*4
        self.masa = []
        self.ancho_pasillo = ancho_pasillo
        self.maximizar = maximizar
        self.verbose = verbose
        self.columnas = 4
        self.x_centro = ((80*4)+self.ancho_pasillo)/2
        self.y_centro = (self.filas*80)/2
    
    def generar_pasajeros(self): #generacion de pasajeros
        for i in range(self.cantidad):
            self.masa.append(round(normalvariate(75,10)))
        
    def generar_poblacion(self):
        poblacion = []
        individuos = self.cantidad
        valor = []
        ids = []
        pasajeros = self.generar_pasajeros()
        
        self.cantidad_pasajeros = individuos
        for i in range(self.cantidad):
            ids.append(i+1)
        for i in range(self.cantidad_pasajeros):
            valor.append(self.masa[i])
        self.pasajeros_id = {i:j for i,j in zip(ids, valor)}
        
        orden_asientos = []
        for i in range(self.poblacion_i):
            movimientos = np.random.randint(0, int(individuos/2))
            desordenar_ids = ids.copy()
            for j in range(movimientos):
                new_posicion = np.random.randint(0, individuos)
                valor_actual = 0
                nuevo_valor = 0
                valor_actual = desordenar_ids[j]
                nuevo_valor = desordenar_ids[new_posicion]
                desordenar_ids[j] = nuevo_valor
                desordenar_ids[new_posicion] = valor_actual
            poblacion.append(desordenar_ids)
            
            area_pasajeros = []
            i=0
            for fila in range(self.filas):  # acomodar en los asientos
                area_pasajeros.append([])
                for columna in range(self.columnas):
                    area_pasajeros[fila].append(self.pasajeros_id[desordenar_ids[i]])
                    i+=1
            # print(area_pasajeros)
            orden_asientos.append(area_pasajeros)
            
        return poblacion,orden_asientos
    
    def calcular_aptitud(self, pasajero_x, pasajero_y):
        aptitud = math.sqrt((pasajero_x - self.x_centro)**2 + (pasajero_y - self.y_centro)**2)
        
        return round(aptitud,5)
    
    def evaluar_poblacion(self, poblacion, distribucion_asientos):    
        
        fitness = []
        flag = True
        area_pasajeros = (self.filas * 4 * 80 * 80) + (self.filas *self.ancho_pasillo *80)
        pasajero_x = 0
        pasajero_y = 0
        total_masa = 0
        for individuo in range(len(poblacion)):
            for i in range(self.filas):
                for j in range(self.columnas):
                    if(distribucion_asientos[individuo][i][j] != 0):
                        x = (i*80) + (self.ancho_pasillo/2)
                        y = (j * 80) 
                        
                        total_masa += distribucion_asientos[individuo][i][j]
                        
                        pasajero_x += x * distribucion_asientos[individuo][i][j]
                        pasajero_y += y * distribucion_asientos[individuo][i][j]
            pasajero_x /= total_masa
            pasajero_y /= total_masa
            
            aptitud = self.calcular_aptitud(pasajero_x, pasajero_y)
            
            conjunto_datos = poblacion[individuo], pasajero_x, pasajero_y,aptitud
            fitness.append(conjunto_datos)
            
        # print(fitness)
        
        return fitness
    
      #metodo de seleccion por tournament
    def seleccion(self, maximizar, fit):
        seleccionados = []
        fitness = fit.copy()       
        fitness.sort(key=lambda x: x[3], reverse=maximizar)
            
        for i in range(int(fitness.__len__()/2)):
            fitness.pop()
        
        for i in range(len(fitness)):
            seleccionados.append(fitness[np.random.randint(0, fitness.__len__())])
        if(len(seleccionados)%2 != 0):
            seleccionados.pop()
        seleccionados.sort(key=lambda x: x[3], reverse=maximizar)
        
        return seleccionados
        
    def buscar_repetidos(self, paquete):
        repetidos = []
        for i in range(len(paquete)):
            for j in range(i+1, len(paquete)):
                if(paquete[i] == paquete[j]):
                    repetido = paquete[i], i
                    repetidos.append(repetido)
        return repetidos
    
    def hacer_paquete_valido(self, paquete):
        faltante = self.no_encontrados(paquete)       
        repetidos = self.buscar_repetidos(paquete)
        if(len(faltante) == 0 and len(repetidos) == 0):
            return paquete
        else:
            for i in range(len(repetidos)):
                paquete[repetidos[i][1]] = faltante[i]
            return paquete

    def no_encontrados(self, paquete):    
        faltantes = [] 
        for i in range(self.cantidad):
            bandera = False
            for j in range(len(paquete)):
                if(paquete[j] == i+1):
                    bandera = True
            if(bandera == False):
                faltantes.append(i+1)        
        return faltantes
    
    #padres provenientes de la seleccion
    def cruzar(self, seleccionados):
        hijo1_head = ""
        hijo1_tail = ""
        hijo2_head = ""
        hijo2_tail = ""
        hijo1 = ""
        hijo2 = ""
        hijos = []
        try:
            padre_ganador = seleccionados[0][0]
            for i in range(int(len(seleccionados)/2)):
                reproduccion = np.random.rand()
                if(reproduccion < self.p_cruza):
                    
                    punto_cruza = np.random.randint(1, len(seleccionados[0][0])-1)
                    hijo1_head = padre_ganador[:punto_cruza]
                    hijo1_tail = seleccionados[i+1][0][punto_cruza:]
                    
                    hijo2_head = seleccionados[i+1][0][:punto_cruza]
                    hijo2_tail = padre_ganador[punto_cruza:]
                    
                    hijo1 = hijo1_head + hijo1_tail
                    hijo2 = hijo2_head + hijo2_tail
                    
                    self.hacer_paquete_valido(hijo1)
                    self.hacer_paquete_valido(hijo2)
                    hijos.append(hijo1)
                    hijos.append(hijo2)
        except:
            print("Error en cruzamiento, no hay suficientes padres")
        return hijos

    #hijos provenientes de la cruza
    def mutacion(self, hijos, pmi, pmg):
        pmi = pmi
        pmg = pmg
        pm = pmi * pmg
        individuos = []
        poblacion_final = []
        for i in range(hijos.__len__()):
            numero_aleatorio = [np.random.rand() for i in range(self.cantidad)]
            individuo = (hijos[i], numero_aleatorio)
            individuos.append(individuo)
        for i in range(hijos.__len__()):
            for j in range(individuos[i][1].__len__()):
                if individuos[i][1][j] < pm:
                    while True:
                        posicion = np.random.randint(0, self.cantidad)
                        if numero_aleatorio != i:
                            break
                    individuo = list(individuos[i][0]) 
                    valor_actual = individuo[j]
                    nuevo_valor = individuo[posicion]
                    individuo[j] = nuevo_valor
                    individuo[posicion] = valor_actual
        for i in range(individuos.__len__()):            
            poblacion_final = poblacion_final + [individuos[i][0]]	
        return poblacion_final
                       
    def agregar_poblacion(self, pob, hijos):
        poblacion = pob.copy()
        poblacion.extend(hijos)
        orden_asientos = []
        for individuo in range(poblacion.__len__()):
            area_pasajeros = []
            columnas = 4
            i=0
            for fila in range(self.filas):  # acomodar en los asientos
                area_pasajeros.append([])
                for columna in range(columnas):
                    pasajero_masa = poblacion[individuo][0][i]
                    area_pasajeros[fila].append(self.pasajeros_id[pasajero_masa])
                    i+=1
            # print(area_pasajeros)
            orden_asientos.append(area_pasajeros)
        return poblacion,orden_asientos        

    def poda(self, poblacion, poblacion_maxima):
        poblacion.sort(key=lambda x: x[3])
        if poblacion.__len__() > poblacion_maxima:
            while poblacion.__len__() > poblacion_maxima:
                poblacion.remove(poblacion[-1])        
        return poblacion

    def ordenar_valores(self, valores, maximizar):
        valores_ordenados = []
        valores_ordenar = []
        for i in range(valores.__len__()):
            valores_ordenar.append((valores[i][3]))
        if maximizar:
            valores_ordenados = sorted(valores_ordenar, key = lambda x:[x], reverse=True)
        else:
            valores_ordenados = sorted(valores_ordenar, key = lambda x:[x]) 
        return valores_ordenados
        
def main(genetico):

    poblacion = []
    generaciones = []
    mejor_individuo = []
    promedio = []
    peor_individuo = []  
    dna = genetico
    asientos = []

    poblacion, asientos = dna.generar_poblacion()
    poblacion = dna.evaluar_poblacion(poblacion,asientos)
    # print("Poblacion inicial: ", len(poblacion))
    for g in range(dna.generaciones):

        valores_antes_poda = dna.mutacion(dna.cruzar(dna.seleccion(True,poblacion)), dna.pmi, dna.pmg)
        valores_antes_poda = dna.evaluar_poblacion(valores_antes_poda, asientos)
        valores_antes_poda, asientos = dna.agregar_poblacion(poblacion, valores_antes_poda)
        poblacion_ordenada = dna.ordenar_valores(valores_antes_poda, dna.maximizar)
        mejor_individuo.append(poblacion_ordenada[0])
        promedio.append(np.mean(poblacion_ordenada))
        peor_individuo.append(poblacion_ordenada[-1])
        poblacion = dna.poda(valores_antes_poda, dna.poblacion_f)
        generaciones.append(poblacion)
        print("Generacion: ", g+1, " de ", dna.generaciones)
        print(poblacion)

    # print("Mejores Individuos: ",mejor_individuo)
    # print("Promedio: ",promedio)
    # print("Peores Individuos: ",peor_individuo)
    
    try:
        rmtree("codigo_genetico\Imagenes")
    except:
        pass 
    os.makedirs("codigo_genetico\Imagenes\Tabla", exist_ok=True)
    os.makedirs("codigo_genetico\Imagenes\Video", exist_ok=True)
    
    plt.plot(mejor_individuo, label="Mejor individuo", color="red", linestyle="-",)
    plt.plot(promedio, label="Promedio", color="blue", linestyle="-",)
    plt.plot(peor_individuo, label="Peor individuo", color="green", linestyle="-")
    plt.legend()
    os.makedirs("codigo_genetico\Imagenes\Grafica/", exist_ok=True)
    plt.savefig("codigo_genetico\Imagenes\Grafica/GraficaHistorial.png")
    plt.close()

    individuos_mostrar = []
    centro_masa_x = []
    centro_masa_y = []
    tamanio = (len(generaciones[0]))
    aptitudes = []
    
    for i in range(len(generaciones)):
        individuos = []
        individuos_peores = []
        masa_x = []
        masa_y = []
        masa_x_peores = []
        masa_y_peores = []
        mejores_aptitudes = []
        peores_aptitudes = []
        
        for j in range(5):
            individuos.append(generaciones[i][j][0]) # individuos
            masa_x.append(generaciones[i][j][1]) # masa x
            masa_y.append(generaciones[i][j][2]) # masa y
            mejores_aptitudes.append(generaciones[i][j][3])
            individuos_peores.append(generaciones[i][(tamanio-5)+j][0]) 
            masa_x_peores.append(generaciones[i][(tamanio-5)+j][1])
            masa_y_peores.append(generaciones[i][(tamanio-5)+j][2])
            peores_aptitudes.append(generaciones[i][(tamanio-5)+j][3])
            
                        
        set_individuo = (individuos +  individuos_peores)
        set_masa_x = (masa_x + masa_x_peores)
        set_masa_y = (masa_y + masa_y_peores)
        set_aptitud = (mejores_aptitudes + peores_aptitudes)
        
        individuos_mostrar.append(set_individuo)
        centro_masa_x.append(set_masa_x)
        centro_masa_y.append(set_masa_y)
        aptitudes.append(set_aptitud)

    for i in range(len(generaciones)):
        fig = go.Figure(data=[go.Table(
            columnwidth = [400,70,70,70],
            header=dict(values=['Individuo',  'Centro de Masa X', 'Centro de Masa Y',"Aptitud"]),
            cells=dict(values=[individuos_mostrar[i],centro_masa_x[i],centro_masa_y[i],aptitudes[i]]))
        ])
        
        #fig.show() #usar para ver las tablas, no recomendado en muchas generaciones
        fig.write_image("codigo_genetico\Imagenes\Tabla/Tabla"+str(i)+".png", width=1500, height=650)
        fig.layout.update(title="Tabla"+str(i))
    img = []   
    for i in range(len(generaciones)):
        img.append(cv2.imread("codigo_genetico\Imagenes\Tabla/Tabla"+str(i)+".png"))
    alto, ancho = img[0].shape[:2]
    video = cv2.VideoWriter('codigo_genetico\Imagenes\Video\mivideo.avi', cv2.VideoWriter_fourcc(*'DIVX'),3, (alto, ancho))
    for i in range(len(img)):
        video.write(img[i]) 
    print("OK")
    interfaz.centro_x.setText("X = "+str(dna.x_centro))
    interfaz.centro_y.setText("Y = "+str(dna.y_centro))
    interfaz.estado.setText("Mejor individuo: " + str(mejor_individuo[-1]))
    interfaz.estado2.setText("Proceso Finalizado")
    # app.closeAllWindows()
  
def send():
    run = True
    try:
        poblacion_inicial = int(interfaz.poblacion_i.text())
        poblacion_final = int(interfaz.poblacion_m.text())
        pmg = float(interfaz.pmg.text())
        pmi = float(interfaz.pmi.text())
        cruzamiento = float(interfaz.pcruza.text())
        maximizar = True
        generaciones = int(interfaz.generaciones.text())
        filas = int(interfaz.filas.text())
        # cantidad = int(interfaz.cantidad.text())
        pasillo = int(interfaz.pasillo.text())
       
        if(filas <=0 or poblacion_inicial < 2 or poblacion_final < 2 or pmg <= 0 or pmi <= 0 or cruzamiento <= 0 or generaciones <= 1):
            interfaz.estado.setText("Error Debes revisar tus datos de entrada")
            interfaz.estado.setStyleSheet("color: red")
            run = False

        if(cruzamiento >= 1 ):
            interfaz.estado.setText("la probabilidad de cruza debe ser menor a 1")
            interfaz.estado.setStyleSheet("color: red")
            run = False
        # if(cantidad<1):
        #     interfaz.estado.setText("No puedes tener 0 pasajeros")
        #     interfaz.setStyleSheet("red");
    
    except:
        interfaz.estado.setText("Los datos no son validos")
        interfaz.estado.setStyleSheet("color: red")
        run = False
           
    if(run):
        interfaz.estado.setText("")
        dna = DNA( poblacion_i = poblacion_inicial, poblacion_f = poblacion_final, pmi = pmi, pmg = pmg, p_cruza = cruzamiento, generaciones = generaciones,  filas = filas, ancho_pasillo = pasillo, maximizar = False, verbose = True)
        main(dna)
        
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("interfaz-problema.ui")
    interfaz.show()
    interfaz.btn_ok.clicked.connect(send)
    sys.exit(app.exec())
    # dna = DNA( poblacion_i = 5, poblacion_f = 10, pmi = 0.9, pmg = 0.9, p_cruza = 0.9, generaciones = 5, filas = 5,ancho_pasillo = 50,  maximizar = False, verbose = True)
    # main(dna)