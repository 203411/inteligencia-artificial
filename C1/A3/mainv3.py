import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, uic
from shutil import rmtree
import plotly.graph_objects as go
import os
import cv2

class DNA():

    cajas_id = []

    def __init__(self, poblacion_i, poblacion_f, pmi, pmg, cruzamiento, costo, espacio, generaciones, espacio_maximo,categoria,cantidad,precio_publico, maximizar=True, verbose = True):
        self.poblacion_i = poblacion_i
        self.poblacion_f = poblacion_f
        self.pmi = pmi
        self.pmg = pmg
        self.cruzamiento = cruzamiento
        self.cantidad_cajas = 0
        self.verbose = verbose
        self.costo = costo
        self.espacio = espacio
        self.generaciones = generaciones
        self.maximizar = maximizar
        self.espacio_maximo = espacio_maximo
        self.categoria = categoria
        self.cantidad = cantidad
        self.precio_publico = precio_publico
        
    def generar_poblacion(self):
        poblacion = []
        individuos = 0
        valor = []
        id = []
        for i in range(len(self.categoria)):
            individuos += self.cantidad[i]
        self.cantidad_cajas = individuos
        for i in range(self.cantidad_cajas):
            id.append(i+1)
        for i in range(len(self.categoria)):
            for j in range(self.cantidad[i]):
                valor.append(i+1)
        self.cajas_id = {i:j for i,j in zip(id,valor)}

        for i in range(self.poblacion_i):
            movimientos = np.random.randint(0, int(individuos/2))
            desordernar_id = id.copy()
            for j in range(movimientos):
                nw_posicion = np.random.randint(0, individuos)
                valor_actual = 0
                nuevo_valor = 0
                valor_actual = desordernar_id[j]
                nuevo_valor = desordernar_id[nw_posicion]
                desordernar_id[j] = nuevo_valor
                desordernar_id[nw_posicion] = valor_actual    
            poblacion.append(desordernar_id)
        return poblacion

    #poblacion proveniente de la generacion aleatoria
    def evaluar_poblacion(self, poblacion):
        fitness = []
        bandera = True
        for i in range(poblacion.__len__()):
            costo_unitario = [] 
            espacio_unitario = []
            espacio_usado = 0
            costo_usado = 0
            valores_usados = []
            valores_descartados = []
            for j in range(poblacion[i].__len__()):
                if(bandera):
                    num_caja = self.cajas_id.get(poblacion[i][j])
                    costo_unitario.append(self.costo[num_caja-1])
                    espacio_unitario.append(self.espacio[num_caja-1])
                    espacio_usado = espacio_usado + espacio_unitario[j]
                    costo_usado = (costo_usado + self.precio_publico[num_caja-1] ) -  self.costo[num_caja-1]
                    valores_usados.append(num_caja) #numero de caja es el valor aleatorio de la caja

                    if(espacio_usado > self.espacio_maximo):
                        valores_usados.pop()
                        valores_descartados.append(num_caja)
                        espacio_usado = espacio_usado - espacio_unitario[j]
                        costo_unitario.pop()
                        espacio_unitario.pop()
                        costo_usado = (costo_usado - self.precio_publico[num_caja-1]) + self.costo[num_caja-1]
                        bandera = False
                else:
                    valores_descartados.append(self.cajas_id.get(poblacion[i][j]))
            
            conjunto_datos = poblacion[i] ,valores_usados, valores_descartados, espacio_usado, costo_usado
            fitness.append(conjunto_datos)
            bandera = True
        return fitness
        
    #metodo de seleccion por tournament
    def seleccion(self, maximizar, fit):
        seleccionados = []
        fitness = fit.copy()
        fitness.sort(key=lambda x: x[4], reverse=maximizar)
        for i in range(int(fitness.__len__()/2)):
            fitness.pop()
        for i in range(len(fitness)):
            seleccionados.append(fitness[np.random.randint(0, fitness.__len__())])
        if(len(seleccionados)%2 != 0):
            seleccionados.pop()
        seleccionados.sort(key=lambda x: x[4], reverse=maximizar)
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
        for i in range(self.cantidad_cajas):
            bandera = False
            for j in range(len(paquete)):
                if(paquete[j] == i+1):
                    bandera = True
            if(bandera == False):
                faltantes.append(i+1)
        return faltantes

    #padres provenientes de la seleccion
    def cruzar(self, seleccionados, cruzamiento):
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
                if(reproduccion < cruzamiento):
                    punto_cruzamiento = np.random.randint(1, len(seleccionados[0][0])-1)
                    hijo1_head = padre_ganador[:punto_cruzamiento]
                    hijo1_tail = seleccionados[i+1][0][punto_cruzamiento:]
                    hijo2_head = seleccionados[i+1][0][:punto_cruzamiento]
                    hijo2_tail = padre_ganador[punto_cruzamiento:]
                
                    #print(seleccionados)
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
            numero_aleatorio = [np.random.rand() for i in range(self.cantidad_cajas)]
            individuo = (hijos[i], numero_aleatorio)
            individuos.append(individuo)
        for i in range(hijos.__len__()):
            for j in range(individuos[i][1].__len__()):
                if individuos[i][1][j] < pm:
                    while True:
                        posicion = np.random.randint(0, self.cantidad_cajas)
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
        poblacion.sort(key=lambda x: x[4], reverse=True)
        if poblacion.__len__() > poblacion_maxima:
            while poblacion.__len__() > poblacion_maxima:
                poblacion.remove(poblacion[-1])
        return poblacion

    def ordenar_valores(self, valores, maximizar):
        valores_ordenados = []
        valores_ordenar = []
        for i in range(valores.__len__()):
            valores_ordenar.append(valores[i][4])
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
    poblacion = dna.evaluar_poblacion(dna.generar_poblacion())
    print("Poblacion inicial: ", len(poblacion))
    for g in range(dna.generaciones):
        valores_antes_poda = dna.mutacion(dna.cruzar(dna.seleccion(True,poblacion),dna.cruzamiento), dna.pmi, dna.pmg)
        valores_antes_poda = dna.evaluar_poblacion(valores_antes_poda)
        valores_antes_poda = dna.agregar_poblacion(poblacion, valores_antes_poda)
        poblacion_ordenada = dna.ordenar_valores(valores_antes_poda, True)
        mejor_individuo.append(poblacion_ordenada[0])
        promedio.append(np.mean(poblacion_ordenada))
        peor_individuo.append(poblacion_ordenada[-1])
        poblacion = dna.poda(valores_antes_poda, dna.poblacion_f)
        generaciones.append(poblacion)
        print("Generacion: ", g+1, " de ", dna.generaciones)
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
    costo_mostrar = []
    espacio_mostrar = []
    usados_mostrar = []
    nousados_mostrar = []
    tamanio = (len(generaciones[0]))
    for i in range(len(generaciones)):
        individuos = []
        costo = []
        espacio = []
        usados = []
        nousados = []
        individuos_peores = []
        costo_peores = []
        espacio_peores = []
        usados_peores = []
        nousados_peores = []
        
        for j in range(5):
            individuos.append(generaciones[i][j][0])
            costo.append(generaciones[i][j][4])
            espacio.append(generaciones[i][j][3])
            usados.append(generaciones[i][j][1])
            nousados.append(generaciones[i][j][2])
            individuos_peores.append(generaciones[i][(tamanio-5)+j][0])
            costo_peores.append(generaciones[i][(tamanio-5)+j][4])
            espacio_peores.append(generaciones[i][(tamanio-5)+j][3])
            usados_peores.append(generaciones[i][(tamanio-5)+j][1])
            nousados_peores.append(generaciones[i][(tamanio-5)+j][2])
        set_individuo = (individuos +  individuos_peores)
        set_costo = (costo + costo_peores)
        set_espacio = (espacio + espacio_peores)
        set_usados = (usados + usados_peores)
        set_nousados = (nousados + nousados_peores)
        individuos_mostrar.append(set_individuo)
        costo_mostrar.append(set_costo)
        espacio_mostrar.append(set_espacio)
        usados_mostrar.append(set_usados)
        nousados_mostrar.append(set_nousados)
    

    for i in range(len(generaciones)):
        fig = go.Figure(data=[go.Table(
            columnwidth = [150,75,50,12,12],
            header=dict(values=['Individuo', 'Usado', 'No Usado', 'Espacio', 'Ganancia']),
            cells=dict(values=[individuos_mostrar[i],usados_mostrar[i],nousados_mostrar[i],espacio_mostrar[i], costo_mostrar[i]]))
        ])
        #fig.show() #usar para ver las tablas, no recomendado en muchas generaciones
        fig.write_image("codigo_genetico\Imagenes\Tabla/Tabla"+str(i)+".png", width=1500, height=1500)
        fig.layout.update(title="Tabla"+str(i))
    img = []   
    for i in range(len(generaciones)):
        img.append(cv2.imread("codigo_genetico\Imagenes\Tabla/Tabla"+str(i)+".png"))
    alto, ancho = img[0].shape[:2]
    video = cv2.VideoWriter('codigo_genetico\Imagenes\Video\mivideo.avi', cv2.VideoWriter_fourcc(*'DIVX'),3, (alto, ancho))
    for i in range(len(img)):
        video.write(img[i]) 
    print("OK")
    interfaz.estado2.setText("Proceso Finalizado")
    app.closeAllWindows()

cantidad = []
categoria = []
precio_publico = []
costo = []
espacio = []
def guardar_valores():
    try:
        cantidad_str = interfaz.cantidad_cajas.text().split(",")
        categoria_str = interfaz.tipo_caja.text().split(",")
        precio_publico_str = interfaz.precio_publico.text().split(",")
        costo_str = interfaz.costo.text().split(",")
        espacio_str = interfaz.espacio.text().split(",")

        if((len(cantidad_str) != len(categoria_str) or len(cantidad_str) != len(precio_publico_str) or len(cantidad_str) != len(costo_str) or len(cantidad_str) != len(espacio_str)) == False):
            for i in range(len(cantidad_str)):

                RuntimeError("Error, valores no guardados")
                cantidad.append(int(cantidad_str[i]))
                categoria.append(categoria_str[i])
                precio_publico.append(int(precio_publico_str[i]))
                costo.append(int(costo_str[i]))
                espacio.append(int(espacio_str[i]))
                interfaz.cantidad_cajas.setText("")
                interfaz.tipo_caja.setText("")
                interfaz.precio_publico.setText("")
                interfaz.costo.setText("")
                interfaz.espacio.setText("")
                interfaz.estado2.setText("Valores guardados")
                interfaz.estado.setText("")
        else:
            interfaz.estado.setText("Los datos no son validos, Asegurese de que los valores sean separados por comas y sean del mismo tamaño")
            interfaz.estado.setStyleSheet("color: red")

    except:
        interfaz.estado.setText("Los datos no son validos")
        interfaz.estado.setStyleSheet("color: red")
        # print(cantidad)
    
   
def send():
 
    run = True
    try:
        poblacion_inicial = int(interfaz.poblacionI.text())
        poblacion_final = int(interfaz.poblacionM.text())
        pmg = float(interfaz.pmg.text())
        pmi = float(interfaz.pmi.text())
        cruzamiento = float(interfaz.pd.text())
        maximizar = True
        generaciones = int(interfaz.generaciones.text())
        tam_contenedor = int(interfaz.tam_contenedor.text())
       
        if(tam_contenedor <=0 or poblacion_inicial < 1 or poblacion_final < 1 or pmg <= 0 or pmi <= 0 or cruzamiento <= 0 or generaciones <= 1):
            interfaz.estado.setText("Error Debes revisar tus datos de entrada")
            interfaz.estado.setStyleSheet("color: red")
            run = False

        if(cruzamiento >= 1 or pmg >= 1 or pmi >= 1):
            interfaz.estado.setText("Error Debes revisar tus datos de entrada")
            interfaz.estado.setStyleSheet("color: red")
            run = False
    
    except:
        interfaz.estado.setText("Los datos no son validos")
        interfaz.estado.setStyleSheet("color: red")
        run = False
           
    if(run):
        interfaz.estado.setText("")
        main(DNA(poblacion_inicial, poblacion_final, pmi, pmg, cruzamiento,costo,espacio, generaciones,tam_contenedor,categoria,cantidad,precio_publico, maximizar))
       
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("interfazv3.ui")
    interfaz.show()
    interfaz.guardar_valores.clicked.connect(guardar_valores)
    interfaz.btn_ok.clicked.connect(send)
    sys.exit(app.exec())