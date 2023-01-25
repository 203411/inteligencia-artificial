import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, uic
from shutil import rmtree
import plotly.graph_objects as go
import os
import cv2

class DNA():
    alumnos_id =[]
    
    def __init__(self, poblacion_i, poblacion_f, pmi, pmg, p_cruza, generaciones, tarifa, espacio, espacio_maximo, alumnos, cantidad, masa,gasolina_kilo, K, maximizar = True, verbose = True):
        self.poblacion_i = poblacion_i
        self.poblacion_f = poblacion_f
        self.pmi = pmi
        self.pmg = pmg
        self.p_cruza = p_cruza
        self.generaciones = generaciones
        self.tarifa = tarifa
        self.espacio = espacio
        self.espacio_maximo = espacio_maximo
        self.alumnos = alumnos
        self.cantidad = cantidad
        self.masa = masa
        self.cantidad_alumnos = 0
        self.gasolina_kilo = gasolina_kilo
        self.K = K
        self.maximizar = maximizar
        self.verbose = verbose
        
    def generar_poblacion(self):
        poblacion = []
        individuos = 0
        valor = []
        ids = []
        for i in range(len(self.alumnos)):
            individuos += self.cantidad[i]
        self.cantidad_alumnos = individuos
        for i in range(self.cantidad_alumnos):
            ids.append(i+1)
        for i in range(len(self.alumnos)):
            for j in range(self.cantidad[i]):
                valor.append(i+1)
        self.alumnos_id = {i:j for i,j in zip(ids, valor)}
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
        print("Poblacion: ",poblacion)
        return poblacion
    
    def evaluar_poblacion(self, poblacion):
        fitness = []
        flag = True
        for i in range(poblacion.__len__()):
            tarifa_unitaria = []
            espacio_unitario = []
            espacio_usado = 0
            tarifa_usada = 0
            valores_usados = []
            valores_descartados = []
            
            for j in range(poblacion[i].__len__()):
                if(flag):
                    pasajero = self.alumnos_id.get(poblacion[i][j])
                    tarifa_unitaria.append(self.tarifa[pasajero-1])
                    espacio_unitario.append(self.espacio[pasajero-1])
                    espacio_usado = espacio_usado + espacio_unitario[j]
                    
                    tarifa_usada += (self.tarifa[pasajero-1] - ((self.K + self.masa[pasajero-1])*self.gasolina_kilo))
                    # print(tarifa_usada)
                    valores_usados.append(pasajero)

                    if(espacio_usado > self.espacio_maximo):
                        valores_usados.pop()
                        valores_descartados.append(pasajero)
                        espacio_usado = espacio_usado - espacio_unitario[j]
                        tarifa_unitaria.pop()
                        espacio_unitario.pop()
                        tarifa_usada -= (self.tarifa[pasajero-1]-(self.K + self.masa[pasajero-1])*self.gasolina_kilo)
                        # print(tarifa_usada)
                        flag = False
                else:
                    valores_descartados.append(self.alumnos_id.get(poblacion[i][j]))
                    
            conjunto_datos = poblacion[i], valores_usados, valores_descartados, espacio_usado, tarifa_usada
            fitness.append(conjunto_datos)
            flag = True
        
        print("Población evaluada: ",fitness)
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
        for i in range(self.cantidad_alumnos):
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
            numero_aleatorio = [np.random.rand() for i in range(self.cantidad_alumnos)]
            individuo = (hijos[i], numero_aleatorio)
            individuos.append(individuo)
        for i in range(hijos.__len__()):
            for j in range(individuos[i][1].__len__()):
                if individuos[i][1][j] < pm:
                    while True:
                        posicion = np.random.randint(0, self.cantidad_alumnos)
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

        valores_antes_poda = dna.mutacion(dna.cruzar(dna.seleccion(True,poblacion)), dna.pmi, dna.pmg)
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
            individuos.append(generaciones[i][j][0]) # individuos
            costo.append(generaciones[i][j][4]) # tarifas
            espacio.append(generaciones[i][j][3]) # espacios
            usados.append(generaciones[i][j][1]) # individuos ocupados
            nousados.append(generaciones[i][j][2]) #individuos no usados
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
        fig.write_image("codigo_genetico\Imagenes\Tabla/Tabla"+str(i)+".png", width=1500, height=700)
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
    # app.closeAllWindows()


cantidad = []
nombres_alumnos = []
tarifas_alumnos = []
masa_alumnos = []
ancho_cadera = []

def guardar_valores():
    try:
        
        alumnos_str = interfaz.nombres.text().split(",")
        tarifas_str = interfaz.tarifas.text().split(",")
        masa_str = interfaz.masas.text().split(",")
        espacio_str = interfaz.espacios.text().split(",")

        if(( len(alumnos_str)  != len(tarifas_str) or  len(masa_str) != len(espacio_str) or len(alumnos_str) != len(tarifas_str)) == False):
            for i in range(len(alumnos_str)):
                

                RuntimeError("Error, valores no guardados")
                
                cantidad.append(1)                
                nombres_alumnos.append(alumnos_str[i])                
                tarifas_alumnos.append(int(tarifas_str[i]))
                masa_alumnos.append(int(masa_str[i]))
                ancho_cadera.append(int(espacio_str[i]))
                interfaz.nombres.setText("")
                interfaz.tarifas.setText("")
                interfaz.masas.setText("")
                interfaz.espacios.setText("")
                interfaz.estado2.setText("Valores guardados")
                interfaz.estado.setText("")
        else:
            interfaz.estado.setText("Los datos no son validos, Asegurese de que los valores sean separados por comas y sean del mismo tamaño")
            interfaz.estado.setStyleSheet("color: red")

    except:
        interfaz.estado.setText("Los datos no son validos")
        interfaz.estado.setStyleSheet("color: red")
        print(cantidad)
            
  
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
        espacio_moto = int(interfaz.espacio_maximo.text())
        precio_gasolina = float(interfaz.gasolina.text())
       
        if(espacio_moto <=0 or poblacion_inicial < 2 or poblacion_final < 2 or pmg <= 0 or pmi <= 0 or cruzamiento <= 0 or generaciones <= 1):
            interfaz.estado.setText("Error Debes revisar tus datos de entrada")
            interfaz.estado.setStyleSheet("color: red")
            run = False

        if(cruzamiento >= 1 ):
            interfaz.estado.setText("la probabilidad de cruza debe ser menor a 1")
            interfaz.estado.setStyleSheet("color: red")
            run = False
    
    except:
        interfaz.estado.setText("Los datos no son validos")
        interfaz.estado.setStyleSheet("color: red")
        run = False
           
    if(run):
        interfaz.estado.setText("")
        
        main( DNA( poblacion_i = poblacion_inicial, poblacion_f = poblacion_final, pmi = pmi, pmg = pmg, p_cruza = cruzamiento, generaciones = generaciones, tarifa = tarifas_alumnos, espacio = ancho_cadera, espacio_maximo = espacio_moto, alumnos = nombres_alumnos, cantidad = cantidad, masa = masa_alumnos,gasolina_kilo = precio_gasolina, K = 400, maximizar = True, verbose = True))
    
    
# nombres = Carlos,Manuel,Sergio,Martha,Carolina,Gustavo,Maria
# masas= 90,85,80,75,80,90,90
# tarifas = 8,8,6,8,6,8,8
# espacios = 60,50,45,45,50,55,55
# precio gasolina= 0.01
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("interfaz.ui")
    interfaz.show()
    interfaz.guardar_valores.clicked.connect(guardar_valores)
    interfaz.btn_ok.clicked.connect(send)
    sys.exit(app.exec())
    # dna = DNA( poblacion_i = 5, poblacion_f = 20, pmi = 0.9, pmg = 0.9, p_cruza = 0.9, generaciones = 5, tarifa = [8,8,6,8,6,8,8], espacio = [60,50,45,45,50,55,55], espacio_maximo = 150, alumnos = ["Carlos","Manuel","Sergio","Martha","Carolina","Gustavo","Maria"], cantidad = [1,1,1,1,1,1,1], masa = [90,85,80,75,80,90,90],gasolina_kilo = 0.01, K = 400, maximizar = True, verbose = True)
    # dna.evaluar_poblacion(dna.generar_poblacion())
    main(dna)