import sys
import os
from PyQt5 import QtWidgets, uic
import math
import numpy as np
from random import uniform, randint




class DNA():
    def __init__(self,poblacion_i,poblacion_m, pmi, pmg, p_cruza,presicion,x_min, x_max, generaciones, maximizar=True, verbose=True):
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
        for i in range(self.poblacion_i):
            individuo = [np.random.randint(0, 2) for i in range(self.calculate_bits(self.calculate_value(self.x_min, self.x_max, self.presicion)))]
            poblacion.append(individuo)
            # print(poblacion)
        return poblacion
    
    '''Función de ejemplo'''
    def fx(self,x):
        return math.cos(math.pi*x)*math.sin(math.pi*x/2)+math.log(x)

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
        a = self.x_max
        delta = (self.x_max - self.x_min) / self.calculate_bits(self.calculate_value(self.x_min, self.x_max, self.presicion))
        valor = 0
        poblacion = poblacion
        fitness = []	
        for i in range(poblacion.__len__()):
            i = self.binary_to_decimal(poblacion.__getitem__(i))
            x = a + i[0] * delta #funcion para calcular el valor de x
            valor = (i.__getitem__(1),x,self.fx(x),i.__getitem__(0))
            fitness.append(valor)
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
        padre_ganador = padres.__getitem__(0).__getitem__(0)
        for i in range(int(len(padres)/2)):
            pc = np.random.rand() #probabilidad de cruza
            if pc <= p_cruza:
                punto_cruza = np.random.randint(1,padres.__getitem__(0).__getitem__(0).__len__())
                # print("\n % de reproduccion: ",pc,"Punto de cruza: ",punto_cruza,"Padre 1: ",padre_ganador,"Padre 2: ",padres[i+1].__getitem__(0)	,"\n")
                hijo1_head = padre_ganador[:punto_cruza]
                hijo1_tail = padres[i+1].__getitem__(0)[punto_cruza:]
                hijo2_head = padres[i+1].__getitem__(0)[:punto_cruza]
                hijo2_tail = padre_ganador[punto_cruza:]
                hijo1 = hijo1_head +""+ hijo1_tail
                hijo2 = hijo2_head +""+ hijo2_tail
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
                    
                    print("individuo: ", individuo)
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
        
        

def main(dna):
    poblacion = []
    generaciones = []
    individuos_before_poda = []
    best_individuo = []
    worst_individuo = []
    promedio = []
    
    poblacion = dna.generate_population() 
    poblacion = dna.evaluate_poblacion(poblacion) 
     
    print("Poblacion inicial: (Generacion 1)",poblacion)
    
    for generacion in range(dna.generaciones):
        individuos_before_poda = dna.mutacion(dna.cruza(dna.selection(dna.maximizar,poblacion), dna.p_cruza ), dna.pmi, dna.pmg)
    print("Poblacion despues de la mutacion: ",individuos_before_poda)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("interfaz.ui")
    interfaz.show()
    main(DNA(poblacion_i=10,poblacion_m=10, pmi=0.1, pmg=0.1, p_cruza=0.4,presicion=0.01,x_min=5, x_max=10, generaciones=10))
    sys.exit(app.exec_())
    