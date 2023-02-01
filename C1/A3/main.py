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
import xlsxwriter

class DNA():
    pasajeros_id =[]
    
    def __init__(self, poblacion_i, poblacion_f, pmi, pmg, p_cruza, generaciones, cantidad, filas,ancho_pasillo, media, desviacion, maximizar, verbose = True):
        self.poblacion_i = poblacion_i
        self.poblacion_f = poblacion_f
        self.pmi = pmi
        self.pmg = pmg
        self.p_cruza = p_cruza
        self.generaciones = generaciones
        self.filas = filas
        self.cantidad = cantidad
        self.masa = []
        self.ancho_pasillo = ancho_pasillo
        self.maximizar = maximizar
        self.verbose = verbose
        self.columnas = 4
        self.x_centro = ((80*4)+self.ancho_pasillo)/2
        self.y_centro = (self.filas*80)/2
        self.media = media
        self.desviacion = desviacion
    
    def generar_pasajeros(self): #generacion de pasajeros
        for i in range(self.cantidad):
            self.masa.append(round(normalvariate(self.media,self.desviacion)))
        
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
        return poblacion
    
    def calcular_aptitud(self, pasajero_x, pasajero_y):
        aptitud = math.sqrt((pasajero_x - self.x_centro)**2 + (pasajero_y - self.y_centro)**2)
        return round(aptitud,5)
    
    def generar_posiciones_y(self):
        y = []
        valor = 40
        for i in range(self.filas):
            y.append(valor)
            valor += 80
        return y
    
    def evaluar_poblacion(self, poblacion):    
        usados = []
        no_usados = []
        distribucion_asientos = []
        for individuo in range(poblacion.__len__()):
            area_pasajeros = []
            
            aux_usados = []
            aux_no_usados = []
            columnas = 4
            i=0
            for fila in range(self.filas):  # acomodar en los asientos
                area_pasajeros.append([])
                for columna in range(self.columnas):
                    if(i<self.cantidad):
                        pasajero_masa = poblacion[individuo][i]
                    if(np.random.rand() < 0.8) and (i<self.cantidad):
                        area_pasajeros[fila].append(self.pasajeros_id[pasajero_masa])
                        aux_usados.append(poblacion[individuo][i])
                    else:
                        area_pasajeros[fila].append(0)
                        if(i<self.cantidad):
                            aux_no_usados.append(poblacion[individuo][i])
                    i+=1
            usados.append(aux_usados)
            no_usados.append(aux_no_usados)
            distribucion_asientos.append(area_pasajeros)
        fitness = []
        x = [40,120,200 + (self.ancho_pasillo),280 + (self.ancho_pasillo)]
        y = self.generar_posiciones_y()
        for individuo in range(len(poblacion)):
            pasajero_x = 0
            pasajero_y = 0
            total_masa = 0
            for i in range(self.filas):
                for j in range(self.columnas):
                    if(distribucion_asientos[individuo][i][j] != 0):                                    
                        total_masa += distribucion_asientos[individuo][i][j]
                        
                        pasajero_x += x[j] * distribucion_asientos[individuo][i][j]
                        pasajero_y += y[i] * distribucion_asientos[individuo][i][j]
                    # print("Individuo ",individuo," ",poblacion[individuo],"\nPosicion X: ",x[j],"\nPosicion Y: ",y[i],"\nMasa: ",distribucion_asientos[individuo][i][j],"\n")
            pasajero_x /= total_masa
            pasajero_y /= total_masa
            # print("Vuelta: ",individuo," X: ",pasajero_x," Y: ",pasajero_y," Total masa: ",total_masa)
            aptitud = self.calcular_aptitud(pasajero_x, pasajero_y)
            # print("Aptitud: ",aptitud,"\n")
            conjunto_datos = poblacion[individuo], usados[individuo],no_usados[individuo],round(pasajero_x,5), round(pasajero_y,5),aptitud
            fitness.append(conjunto_datos)       
        return fitness

    def seleccion(self, maximizar, fit):
        seleccionados = []
        fitness = fit.copy()       
        fitness.sort(key=lambda x: x[5])
            
        for i in range(int(fitness.__len__()/2)):
            fitness.pop() # Se elimina a la mitad de la población con peor fitness y que no son utilizados
        
        for i in range(len(fitness)):
            seleccionados.append(fitness[np.random.randint(0, fitness.__len__())])
        if(len(seleccionados)%2 != 0):
            seleccionados.pop()
        seleccionados.sort(key=lambda x: x[5])
        
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
        
        return poblacion        

    def poda(self, poblacion, poblacion_maxima):
        poblacion.sort(key=lambda x: x[5])
        if poblacion.__len__() > poblacion_maxima:
            while poblacion.__len__() > poblacion_maxima:
                poblacion.pop() 
        # Eliminar individuos repetidos
        for individuo in poblacion:
            while(poblacion.count(individuo) > 1):
                poblacion.remove(individuo)
        
        return poblacion

    def ordenar_valores(self, valores, maximizar):
        valores_ordenados = []
        valores_ordenar = []
        for i in range(valores.__len__()):
            valores_ordenar.append((valores[i][5]))
        valores_ordenados = sorted(valores_ordenar, key = lambda x:[x]) 
        return valores_ordenados

def crear_excel(individuos_mostrar,usados_mostrar, no_usados_mostrar, centro_masa_x, centro_masa_y, aptitudes):
    os.makedirs("codigo_genetico\Excel/", exist_ok=True)
    libro = xlsxwriter.Workbook('codigo_genetico\Excel\AG.xlsx')
    hoja = libro.add_worksheet()
    hoja.write(0,0,"Generacion")
    hoja.write(0,1,"Individuo")
    hoja.write(0,2,"Usados")
    hoja.write(0,3,"No Usados")
    hoja.write(0,4,"Centro de Masa X")
    hoja.write(0,5,"Centro de Masa Y")
    hoja.write(0,6,"Aptitud")
    
    row = 1
    for g in range(individuos_mostrar.__len__()):
        for i in range(individuos_mostrar[g].__len__()):
            hoja.write(row,0,str(g+1))
            hoja.write(row,1,str(individuos_mostrar[g][i]))
            hoja.write(row,2,str(usados_mostrar[g][i]))
            hoja.write(row,3,str(no_usados_mostrar[g][i]))
            hoja.write(row,4,centro_masa_x[g][i])
            hoja.write(row,5,centro_masa_y[g][i])
            hoja.write(row,6,aptitudes[g][i])
            row+=1
        row+=1
    libro.close()

def crear_graficas_dinamicas(generaciones):
    for i in range(len(generaciones)):
        centro_x=[]
        centro_y=[]
        for j in range(generaciones[i].__len__()):
            centro_x.append(generaciones[i][j][3])
            centro_y.append(generaciones[i][j][4])
        
        fig,aux=plt.subplots()
        x=np.array(centro_x)
        y=np.array(centro_y)
        plt.grid()
        plt.scatter(x,y,label='Individuos')
        x_centro=np.array([dna.x_centro])
        y_centro=np.array([dna.y_centro])
        plt.scatter(x_centro,y_centro,label='Centro de masa')
        aux.set_title(f'Generacion: {i+1}',fontdict={'fontsize':20})
        aux.set_xlabel('X',fontdict={'fontsize':15})
        aux.set_ylabel('Y',fontdict={'fontsize':1})
        aux.legend(loc='upper right')
        os.makedirs("codigo_genetico/Imagenes/GraficaUnitaria", exist_ok=True)
        plt.savefig(f'codigo_genetico/Imagenes/GraficaUnitaria/Generacion{i+1}')
        plt.close()
    
    img = []
    for i in range(len(generaciones)):
        img.append(cv2.imread("codigo_genetico/Imagenes/GraficaUnitaria/Generacion"+str(i+1)+".png"))
    alto, ancho, canales = img[0].shape[:3]
    video = cv2.VideoWriter('codigo_genetico/Imagenes/Video/Graficas.avi', cv2.VideoWriter_fourcc(*'DIVX'), 3, (ancho, alto))
    for i in range(len(img)):
        video.write(img[i])
    print("Graficas dinamicas creadas")
    

def crear_graficas_estaticas(generaciones,centro_masa_x,centro_masa_y,dna):
    for i in range(len(generaciones)):
        centro_x=[]
        centro_y=[]
        for j in range(generaciones[i].__len__()):
            centro_x.append(generaciones[i][j][3])
            centro_y.append(generaciones[i][j][4])
        
        fig,aux=plt.subplots()
        x=np.array(centro_x)
        y=np.array(centro_y)
        # plt.xlim(min(centro_masa_x[0])-1,max(centro_masa_x[0])+1)
        # plt.ylim(min(centro_masa_y[0])-1,max(centro_masa_y[0])+1)
        plt.xlim(dna.x_centro-55, dna.x_centro+55)
        plt.ylim(dna.y_centro-45, dna.y_centro+45)
        plt.grid(linewidth=0.5, color= "gray")
        plt.scatter(x,y,label='Individuos', s = 10)
        x_centro=np.array([dna.x_centro])
        y_centro=np.array([dna.y_centro])
        plt.scatter(x_centro,y_centro,label='Centro de masa')
        aux.set_title(f'Generacion: {i+1}',fontdict={'fontsize':20})
        aux.set_xlabel('X',fontdict={'fontsize':15})
        aux.set_ylabel('Y',fontdict={'fontsize':1})
        aux.legend(loc='upper right')
        os.makedirs("codigo_genetico/Imagenes/Ubicacion", exist_ok=True)
        plt.savefig(f'codigo_genetico/Imagenes/Ubicacion/Generacion{i+1}')
        plt.close()
    
    img = []
    for i in range(len(generaciones)):
        img.append(cv2.imread("codigo_genetico/Imagenes/Ubicacion/Generacion"+str(i+1)+".png"))
    alto, ancho, canales = img[0].shape[:3]
    video = cv2.VideoWriter('codigo_genetico/Imagenes/Video/Ubicacion.avi', cv2.VideoWriter_fourcc(*'DIVX'), 3, (ancho, alto))
    for i in range(len(img)):
        video.write(img[i])
    print("Graficas y videos creados")
   
   
def crear_tablas(generaciones,individuos_mostrar,usados_mostrar, no_usados_mostrar, centro_masa_x, centro_masa_y, aptitudes):
    for i in range(len(generaciones)):
        fig = go.Figure(data=[go.Table(
            columnwidth = [400,70,70,80,80,70],
            header=dict(values=['Individuo',"Usados","No usados",  'Centro de Masa X', 'Centro de Masa Y',"Aptitud"]),
            cells=dict(values=[individuos_mostrar[i],usados_mostrar[i],no_usados_mostrar[i],centro_masa_x[i],centro_masa_y[i],aptitudes[i]]))
        ])
        
        #fig.show() #usar para ver las tablas, no recomendado en muchas generaciones
        fig.write_image("codigo_genetico\Imagenes\Tabla/Tabla"+str(i)+".png", width=1500, height=650)
        fig.layout.update(title="Tabla"+str(i))
    # img = []   
    # for i in range(len(generaciones)):
    #     img.append(cv2.imread("codigo_genetico\Imagenes\Tabla/Tabla"+str(i)+".png"))
    # alto, ancho = img[0].shape[:2]
    # video = cv2.VideoWriter('codigo_genetico\Imagenes\Video\mivideo.avi', cv2.VideoWriter_fourcc(*'DIVX'),3, (alto, ancho))
    # for i in range(len(img)):
    #     video.write(img[i]) 
    print("Tablas creadas")
        
def main(genetico):

    poblacion = []
    generaciones = []
    mejor_individuo = []
    promedio = []
    peor_individuo = []  
    dna = genetico
    asientos = []
    usados = []
    no_usados = []

    poblacion = dna.generar_poblacion()
    poblacion = dna.evaluar_poblacion(poblacion)
    # print("Poblacion inicial: ", len(poblacion))
    for g in range(dna.generaciones):

        valores_antes_poda = dna.mutacion(dna.cruzar(dna.seleccion(dna.maximizar,poblacion)), dna.pmi, dna.pmg)
        valores_antes_poda = dna.evaluar_poblacion(valores_antes_poda)
        valores_antes_poda = dna.agregar_poblacion(poblacion, valores_antes_poda)
        poblacion_ordenada = dna.ordenar_valores(valores_antes_poda, dna.maximizar)
        mejor_individuo.append(poblacion_ordenada[0])
        promedio.append(np.mean(poblacion_ordenada))
        peor_individuo.append(poblacion_ordenada[-1])
        poblacion = dna.poda(valores_antes_poda, dna.poblacion_f)
        generaciones.append(poblacion)
        print("Generacion: ", g+1, " de ", dna.generaciones)
        # print(poblacion)
    
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
    usados_mostrar = []
    no_usados_mostrar = []
    tamanio = (len(generaciones[0]))
    aptitudes = []
    
    for i in range(len(generaciones)):
        individuos = []
        individuos_peores = []
        masa_x = []
        masa_y = []
        masa_x_peores = []
        masa_y_peores = []
        usados = []
        no_usados = []
        mejores_aptitudes = []
        peores_aptitudes = []
        usados_peores = []
        no_usados_peores = []
        
        # Tomar a los 5 mejores y los 5 peores de cada generacion
        for j in range(5):
            individuos.append(generaciones[i][j][0]) # individuos
            usados.append(generaciones[i][j][1]) # usados
            no_usados.append(generaciones[i][j][2]) # no usados
            masa_x.append(generaciones[i][j][3]) # masa x
            masa_y.append(generaciones[i][j][4]) # masa y
            mejores_aptitudes.append(generaciones[i][j][5])
            individuos_peores.append(generaciones[i][(tamanio-5)+j][0]) 
            usados_peores.append(generaciones[i][(tamanio-5)+j][1])
            no_usados_peores.append(generaciones[i][(tamanio-5)+j][2])
            masa_x_peores.append(generaciones[i][(tamanio-5)+j][3])
            masa_y_peores.append(generaciones[i][(tamanio-5)+j][4])
            peores_aptitudes.append(generaciones[i][(tamanio-5)+j][5])
                        
        set_individuo = (individuos +  individuos_peores)
        set_masa_x = (masa_x + masa_x_peores)
        set_masa_y = (masa_y + masa_y_peores)
        set_aptitud = (mejores_aptitudes + peores_aptitudes)
        set_usados = (usados + usados_peores)
        set_no_usados = (no_usados + no_usados_peores)
        
        individuos_mostrar.append(set_individuo)
        centro_masa_x.append(set_masa_x)
        centro_masa_y.append(set_masa_y)
        aptitudes.append(set_aptitud)
        usados_mostrar.append(set_usados)
        no_usados_mostrar.append(set_no_usados)

    crear_excel(individuos_mostrar,usados_mostrar, no_usados_mostrar, centro_masa_x, centro_masa_y, aptitudes)   
    # crear_graficas_dinamicas(generaciones)
    crear_graficas_estaticas(generaciones,centro_masa_x, centro_masa_y,dna)
    # crear_tablas(generaciones)
    
    interfaz.centro_x.setText("X = "+str(centro_masa_x[-1][0]))
    interfaz.centro_y.setText("Y = "+str(centro_masa_y[-1][0]))
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
        generaciones = int(interfaz.generaciones.text())
        filas = int(interfaz.filas.text())
        cantidad = int(interfaz.cantidad.text())
        pasillo = int(interfaz.pasillo.text())
        media = int(interfaz.media.text())
        desviacion = int(interfaz.desviacion.text())
        
        if(desviacion == ""):
            desviacion = 10
        if(media == ""):
            media = 75
       
        if(filas <=0 or poblacion_inicial < 2 or poblacion_final < 2 or pmg <= 0 or pmi <= 0 or cruzamiento <= 0 or generaciones <= 1):
            interfaz.estado.setText("Error Debes revisar tus datos de entrada")
            interfaz.estado.setStyleSheet("color: red")
            run = False

        if(cruzamiento >= 1 ):
            interfaz.estado.setText("la probabilidad de cruza debe ser menor a 1")
            interfaz.estado.setStyleSheet("color: red")
            run = False
        if(cantidad<1):
            interfaz.estado.setText("No puedes tener 0 pasajeros")
            interfaz.setStyleSheet("red");
    
    except:
        interfaz.estado.setText("Los datos no son validos")
        interfaz.estado.setStyleSheet("color: red")
        run = False
           
    if(run):
        interfaz.estado.setText("")
        dna = DNA( poblacion_i = poblacion_inicial, poblacion_f = poblacion_final, pmi = pmi, pmg = pmg, p_cruza = cruzamiento, generaciones = generaciones,cantidad = cantidad,  filas = filas, ancho_pasillo = pasillo, media = media, desviacion = desviacion, maximizar = False, verbose = True)
        main(dna)
        
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("interfaz.ui")
    interfaz.show()
    interfaz.btn_ok.clicked.connect(send)
    sys.exit(app.exec())
    # dna = DNA( poblacion_i = 50, poblacion_f = 100, pmi = 0.9, pmg = 0.9, p_cruza = 0.9, generaciones = 100,cantidad = 100, filas = 20,ancho_pasillo = 100, media = 75, desviacion = 10,  maximizar = False, verbose = True)
    # main(dna)