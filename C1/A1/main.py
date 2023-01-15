import sys
import os
from PyQt5 import QtWidgets, uic
import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform, randint
from shutil import rmtree
import cv2
class DNA():
    def __init__(self,poblacion_i,poblacion_m, pmi, pmg, p_cruza,presicion,x_max, x_min, generaciones, maximizar=True, verbose=True):
        self.poblacion_i = poblacion_i
        self.poblacion_m = poblacion_m
        self.pmi = pmi
        self.pmg = pmg
        self.p_cruza = p_cruza
        self.presicion = presicion
        self.x_min = x_min
        self.x_max = x_max
        self.generaciones = generaciones
        self.maximizar = maximizar
        self.verbose = verbose
        
    '''Calcula el numero de puntos'''
    def calculate_value(self, x_min, x_max, presicion):
        valor_maximo = ((x_max-x_min)/presicion)+1
        # print("Valor maximo: ",valor_maximo)
        return valor_maximo
    
    '''Calcula el numero de bits que se necesitan para representar el valor maximo que puede tomar la variable'''
    def calculate_bits(self, calculo_valor):
        bits = math.ceil(math.log(calculo_valor,2))
        return bits
    
    '''Crea una poblacion de individuos, cada individuo es una lista de numeros binarios
    El numero de bits de cada individuo es el numero de bits que se necesitan para representar el valor maximo que puede tomar la variable
    '''
    def generate_population(self):
        '''array de individuos'''
        poblacion = []
        i = 0
        for i in range(self.poblacion_i):
            individuo = [np.random.randint(0, 2) for i in range(self.calculate_bits(self.calculate_value(self.x_min, self.x_max, self.presicion)))]
            # print(individuo)
            poblacion.append(individuo)
            # print(poblacion)
        return poblacion
    
    
    '''Función de ejemplo'''
    def fx(self,x):
        # return math.cos(math.pi*x)*math.sin(math.pi*x/2)+math.log(x)
        return (((x**3)+(x**2)-(2*x) + 1)/x**2) * math.sin(5*x)

    '''Convertir binario a decimal '''   
    def binary_to_decimal(self,individuo):
        decimal = 0
        cadena = ""
        for i in range(len(individuo)):
            cadena += str(individuo[i])    
        for posicion, digito_string in enumerate(cadena[::-1]):
            decimal += int(digito_string) * 2 ** posicion
        return(decimal, cadena)

    
    def evaluate_poblacion(self, poblacion):
        '''Evalua la poblacion de la generacion aleatoria'''
        x = 0.0
        a = self.x_min
        delta = (self.x_max - self.x_min) / self.calculate_bits(self.calculate_value(self.x_min, self.x_max, self.presicion))
        valor = 0
        poblacion = poblacion
        fitness = []	
        for i in range(poblacion.__len__()):
            i = self.binary_to_decimal(poblacion.__getitem__(i))
            x = a + (i[0] * delta) #funcion para calcular el valor de x
            print(x)
            valor = (i.__getitem__(1),x,self.fx(x),i.__getitem__(0))
            fitness.append(valor)
        # print(fitness)
        return fitness
    
    def selection(self, maximizar, valor):
        '''Selecciona los individuos con mejor fitness'''
        fitness = valor.copy()
        padres = []
        fitness.sort(key=lambda x: x[2], reverse=maximizar)
        for i in range(int(len(fitness)/2)):
            fitness.pop()
        for i in range(int(len(fitness))):
            padres.append(fitness[np.random.randint(0, len(fitness))])
        if padres.__len__() % 2 != 0:
            padres.pop()
        padres.sort(key=lambda x: x[2], reverse=maximizar)     
        # print(padres)  
        return padres
    
    def cruza(self, padres,p_cruza):
        '''Cruza los individuos seleccionados como futuros padres :D'''      
        hijo1_head = ""
        hijo1_tail = ""
        hijo2_head = ""
        hijo2_tail = ""
        hijo1 = ""
        hijo2 = ""
        hijos = []    
        
        punto_cruza =int(padre_ganador.__len__()/2)
        # for i in range((padres.__len__())):
        padre_ganador = padres.__getitem__(0).__getitem__(0)
        for i in range(int(len(padres)-2)):
            pc = np.random.rand() #probabilidad de cruza
            if pc <= p_cruza:
                # punto_cruza = np.random.randint(1,padres.__getitem__(0).__getitem__(0).__len__())
                # print("\n % de reproduccion: ",pc,"Punto de cruza: ",punto_cruza,"Padre 1: ",padre_ganador,"Padre 2: ",padres[i+1].__getitem__(0)	,"\n")
                hijo1_head = padre_ganador[:punto_cruza]
                hijo1_tail = padres[i+1].__getitem__(0)[punto_cruza:]
                hijo2_head = padres[i+1].__getitem__(0)[:punto_cruza]
                hijo2_tail = padre_ganador[punto_cruza:]
                hijo1 = hijo1_head +""+ hijo1_tail
                hijo2 = hijo2_head +""+ hijo2_tail
                # print("Hijo 1: ",hijo1,"Hijo 2: ",hijo2)
                hijos.append(hijo1)
                hijos.append(hijo2)
            else:
              # print("\n % de reproduccion: ",pc)
                pass
            
            
    # print("Hijos: ",hijos)
        return hijos
    
    def mutacion(self, hijos, pmi, pmg):
        pmi = pmi
        pmg = pmg
        pm = pmi * pmg
        individuos = []
        
        poblacion_final = []
        for i in range(hijos.__len__()):
            numero_aleatorio = [np.random.rand() for i in range(self.calculate_bits(self.calculate_value(self.x_min, self.x_max, self.presicion)))]
            individuo = (hijos[i], numero_aleatorio)
            individuos.append(individuo)
    
        for i in range(hijos.__len__()):
            for j in range(individuos[i].__getitem__(1).__len__()):
                if individuos[i].__getitem__(1)[j] < pm:
                    individuo = list(individuos[i].__getitem__(0))
                    
                    # print("individuo: ", individuo)
                    if individuo[j] == "0":
                        individuo[j] = "1"
                        individuoMutado = "".join(individuo)
                        individuos[i] = (individuoMutado, individuos[i].__getitem__(1))
                        
                    else:
                        individuo[j] = "0"
                        individuoMutado = "".join(individuo)
                        individuos[i] = (individuoMutado, individuos[i].__getitem__(1))
                        

        for i in range(individuos.__len__()):            
            poblacion_final.append(individuos[i].__getitem__(0))
        
        return poblacion_final
    
    
    def agregar_poblacion(self, poblacion, hijos):
        poblacion.extend(hijos)
        return poblacion

    def limpieza(self, mutados, pob):
        poblacion = pob.copy()
        x = 0.0
        a = self.x_min
        delta = (self.x_max - self.x_min) / self.calculate_bits(self.calculate_value(self.x_min, self.x_max, self.presicion))
        valor = 0
        decimal = 0
        poblacion_nueva = []
        individuo_completo = []
        
        for i in range(mutados.__len__()):
            for posicion, digito_string in enumerate(mutados[i][::-1]):
                decimal += int(digito_string) * 2 ** posicion
                
            x = a + decimal * delta    
            individuo_completo = (mutados[i], x, self.fx(x), decimal)
            poblacion_nueva.append(individuo_completo)
            decimal = 0
        #print(poblacion) 
        
        poblacion_final = self.agregar_poblacion(poblacion_nueva, poblacion)
        j=0
        k = 0
        
        for i in range(len(poblacion_final)):
            if (poblacion_final[j].__getitem__(1) > self.x_max or poblacion_final[j].__getitem__(1) < self.x_min):
                poblacion_final.remove(poblacion_final[j])
                j = j - 1
            j=j+1
        
        
        return poblacion_final
    
    def poda(self, poblacion, poblacion_maxima):
        
        if len(poblacion) > poblacion_maxima:
            while len(poblacion) > poblacion_maxima:
                poblacion.remove(poblacion[-1])
            #print(poblacion)
        
        return poblacion




    def ordenar_valores(self, valores, maximizar):
        valores_ordenados = []
        valores_ordenar = []
        
        for i in range(valores.__len__()):
            valores_ordenar.append(valores.__getitem__(i).__getitem__(2))
            
        if maximizar:
            valores_ordenados = sorted(valores_ordenar, key = lambda x:float(x), reverse=True)
        else:
            valores_ordenados = sorted(valores_ordenar, key = lambda x:float(x)) 
        return valores_ordenados

        
        

def main(dna, interfaz):
    poblacion = []
    generaciones = []
    individuos_before_poda = []
    mejor_individuo = []
    peor_individuo = []
    promedio = []
    
    poblacion = dna.evaluate_poblacion(dna.generate_population()) 
     
    print("Poblacion inicial: (Generacion 1)",poblacion)
    
    for generacion in range(dna.generaciones):
        individuos_before_poda = dna.limpieza(dna.mutacion(dna.cruza(dna.selection(dna.maximizar,poblacion), dna.p_cruza ), dna.pmi, dna.pmg),poblacion)
        
        poblacion_ordenada = dna.ordenar_valores(individuos_before_poda, dna.maximizar)
        
        mejor_individuo.append(poblacion_ordenada[0])
        promedio.append(np.mean(poblacion_ordenada))
        peor_individuo.append(poblacion_ordenada[-1])
        
        # print("Mejor individuo: ", mejor_individuo)
        # print("Promedio: ", promedio)
        # print("Peor individuo: ", peor_individuo)
        
        individuos_before_poda.sort(key=lambda x: float(x.__getitem__(2)), reverse=dna.maximizar)
        poblacion = dna.poda(individuos_before_poda, dna.poblacion_m)
        generaciones.append(poblacion)
        
    interfaz.estado2.setText("Mejor individuo: " + str(mejor_individuo[-1]))
    for i in range(generaciones.__len__()):
        print("Generacion: ",i+1," ",generaciones[i])
    
    plt.plot(mejor_individuo, label="Mejor individuo", color="red", linestyle="-",)
    plt.plot(promedio, label="Promedio", color="blue", linestyle="-",)
    plt.plot(peor_individuo, label="Peor individuo", color="green", linestyle="-")
    plt.legend()
    os.makedirs("codigo_genetico\Imagenes\GraficaHistorial/", exist_ok=True)
    plt.savefig("codigo_genetico\Imagenes\GraficaHistorial/GraficaHistorial.png")
    plt.close()
    
    try:
        rmtree("codigo_genetico\Imagenes\graficasUnitarias/")
    except:
        pass
    finally:
        os.makedirs("codigo_genetico\Imagenes\graficasUnitarias", exist_ok=True)
    os.makedirs("codigo_genetico\Imagenes\short/", exist_ok=True)
    for i in range(len(generaciones)):
        listaX = []
        listaY = []
        for j in range(len(generaciones[i])):
            listaX.append(generaciones[i].__getitem__(j).__getitem__(1))
            listaY.append(generaciones[i].__getitem__(j).__getitem__(2))

        plt.title("Generacion: " + str(i+1))
        plt.scatter(listaX, listaY)
        plt.xlim(dna.x_min-1,dna.x_max+1)
        plt.ylim(listaY[-1],listaY[0])
        plt.savefig("codigo_genetico\Imagenes\graficasUnitarias/generacion"+str(i+1)+".png")
        plt.savefig("codigo_genetico\Imagenes\short/generacion"+str(i+1)+".png")
        plt.close()
        
    img = []   
    for i in range(len(generaciones)):
        img.append(cv2.imread("codigo_genetico\Imagenes\short\generacion"+str(i+1)+".png"))

    alto, ancho = img[0].shape[:2]

    video = cv2.VideoWriter('codigo_genetico\Imagenes\mivideo.avi', cv2.VideoWriter_fourcc(*'DIVX'), 3, (ancho, alto))

    for i in range(len(img)):
        video.write(img[i])
    rmtree("codigo_genetico\Imagenes\short/")
    print("OK")
    
    # interfaz.estado2.setText("Proceso Finalizado")
    # app.closeAllWindows()
    
    
    ##dna.evaluar_poblacion(dna.generar_poblacion()) ,dna.poblacion_maxima)



def send():
    
    run = True
    try:
        poblacion_inicial = int(interfaz.poblacion_i.text())
        
        poblacion_final = int(interfaz.poblacion_m.text())
        # poblacion_inicial = int(np.random.randint(2,poblacion_final-1))
        print(poblacion_inicial)
        presicion = float(interfaz.presicion.text())
        pmg = float(interfaz.pmg.text())
        pmi = float(interfaz.pmi.text())
        p_cruza = float(interfaz.pcruza.text())
        x_max = int(interfaz.xmax.text())
        x_min = int(interfaz.xmin.text())
        maximizar = bool(interfaz.maximizar.isChecked())
        generaciones = int(interfaz.generaciones.text())
    
    
        if(poblacion_inicial < 1 or poblacion_final < 1 or presicion <= 0 or pmg <= 0 or pmi <= 0 or p_cruza <= 0 or generaciones <= 1):
            interfaz.estado.setText("Error Debes revisar tus datos de entrada")
            interfaz.estado.setStyleSheet("color: red")
            run = False

        if(x_min > x_max):
            interfaz.estado.setText("Error XMin no debe ser mayor a XMax")
            interfaz.estado.setStyleSheet("color: red")
            run = False

        if( p_cruza >= 1):
            interfaz.estado.setText("Error Las probabilidades deben ser menores a 1")
            interfaz.estado.setStyleSheet("color: red")
            run = False
    
    except:
        interfaz.estado.setText("Los datos no son validos")
        interfaz.estado.setStyleSheet("color: red")
        run = False
           
    if(run):
        interfaz.estado.setText("")
        main(DNA(poblacion_i = poblacion_inicial, poblacion_m = poblacion_final, presicion = presicion, pmg = pmg, pmi = pmi, p_cruza = p_cruza, x_max = x_max, x_min = x_min,  generaciones = generaciones, maximizar = maximizar),interfaz)

    
    

if __name__ == "__main__": 
    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("interfaz.ui")
    interfaz.show()
    interfaz.btn_ok.clicked.connect(send)
    
    sys.exit(app.exec())
    