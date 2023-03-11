import numpy as np
import pandas as pd
from operator import itemgetter, attrgetter

class DNA():
    encontrar_valido = False
    tabla_habilidades = []
    
    def __init__(self, habilidades):
        self.habilidades = habilidades

    def abrir_csv(self, archivo):
        genes = pd.read_csv(archivo)
        return genes
    
    def generar_poblacion(self,genes):
        poblacion = []
        individuos = []
        genes = genes.values.tolist()
        self.tabla_habilidades = genes
        for i in range(len(genes)):
            individuos.append(i+1)
        for i in range(10):
            movimientos = np.random.randint(0, int(len(individuos)/2))
            desordernar_id = individuos.copy()
            for j in range(movimientos):
                nw_posicion = np.random.randint(0, len(individuos))
                valor_actual = 0
                nuevo_valor = 0
                valor_actual = desordernar_id[j]
                nuevo_valor = desordernar_id[nw_posicion]
                desordernar_id[j] = nuevo_valor
                desordernar_id[nw_posicion] = valor_actual    
            poblacion.append(desordernar_id)
        return poblacion

    def evaluar_poblacion(self, poblacion):
        poblacion_evaluada = []
        for individuo in poblacion:
            habilidades_requeridas = self.habilidades.copy()
            personas_en_equipo = []
            salario_individual = []
            habilidades_desempenadas = []
            habilidad_desempenada = []
            cantidad_personas_equipo = 0
            costo_mensual_equipo = 0
            for gen in individuo:
                valor_csv = self.tabla_habilidades[gen-1]    
                if habilidades_requeridas.__len__() > 0: 
                    cantidad_personas_equipo += 1
                    personas_en_equipo.append(valor_csv[0])
                    try:
                        for i in range(4):
                            habilidades_requeridas.remove(valor_csv[i+1])
                    except:
                        pass
                    costo_mensual_equipo += valor_csv[5]
                    salario_individual.append(valor_csv[5])
                    habilidad_desempenada = [valor_csv[1],valor_csv[2],valor_csv[3],valor_csv[4]]
                    habilidades_desempenadas.append(habilidad_desempenada)
                else:
                    self.encontrar_valido = True
                    break
            individuo_completo = [individuo, cantidad_personas_equipo,personas_en_equipo, costo_mensual_equipo, salario_individual, habilidades_desempenadas]
            # print(individuo_completo)
            poblacion_evaluada.append(individuo_completo)
        return poblacion_evaluada

    def is_valid(self):
        return self.encontrar_valido
        
    def seleccion(self, fit):
        seleccionados = []
        fitness = fit.copy()
        fitness = sorted(fit, key=itemgetter(1,3))
        for i in range(int(fitness.__len__()/2)):
            fitness.pop()
        for i in range(len(fitness)):
            seleccionados.append(fitness[np.random.randint(0, fitness.__len__())])
        if(len(seleccionados)%2 != 0):
            seleccionados.pop()
        seleccionados = sorted(fit, key=itemgetter(1,3))
        return seleccionados
        
    def buscar_repetidos(self, individuo):
        repetidos = []
        for i in range(len(individuo)):
            for j in range(i+1, len(individuo)):
                if(individuo[i] == individuo[j]):
                    repetido = individuo[i], i
                    repetidos.append(repetido)
        return repetidos
    
    def hacer_grupo_valido(self, individuo):
        faltante = self.no_encontrados(individuo)
        repetidos = self.buscar_repetidos(individuo)
        if(len(faltante) == 0 and len(repetidos) == 0):
            return individuo
        else:
            for i in range(len(repetidos)):
                individuo[repetidos[i][1]] = faltante[i]
            return individuo

    def no_encontrados(self, individuo):    
        faltantes = [] 
        for i in range(self.tabla_habilidades.__len__()):
            bandera = False
            for j in range(len(individuo)):
                if(individuo[j] == i+1):
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
        padre_ganador = seleccionados[0][0]
        for i in range(int(len(seleccionados)/2)):
            reproduccion = np.random.rand()
            if(reproduccion < cruzamiento):
                punto_cruzamiento = np.random.randint(1, len(seleccionados[0][0])-1)
                hijo1_head = padre_ganador[:punto_cruzamiento]
                hijo1_tail = seleccionados[i+1][0][punto_cruzamiento:]
                hijo2_head = seleccionados[i+1][0][:punto_cruzamiento]
                hijo2_tail = padre_ganador[punto_cruzamiento:]
                hijo1 = hijo1_head + hijo1_tail
                hijo2 = hijo2_head + hijo2_tail
                self.hacer_grupo_valido(hijo1)
                self.hacer_grupo_valido(hijo2)
                hijos.append(hijo1)
                hijos.append(hijo2)
        return hijos

    def mutacion(self, hijos, pmi, pmg):
        pmi = pmi
        pmg = pmg
        pm = pmi * pmg
        individuos = []
        poblacion_final = []
        for i in range(hijos.__len__()):
            numero_aleatorio = [np.random.rand() for i in range(self.tabla_habilidades.__len__())]
            individuo = (hijos[i], numero_aleatorio)
            individuos.append(individuo)
        for i in range(hijos.__len__()):
            individuos_mutar = individuos[i][0]
            for j in range(individuos[i][1].__len__()):
                if individuos[i][1][j] < pm:
                    while True:
                        posicion = np.random.randint(0, self.tabla_habilidades.__len__())
                        if posicion != i:
                            break   
                    individuo = individuos_mutar
                    valor_actual = individuo[j]
                    nuevo_valor = individuo[posicion]
                    individuo[j] = nuevo_valor
                    individuo[posicion] = valor_actual
                    individuos_mutar = individuo
        for i in range(individuos.__len__()):            
            poblacion_final = poblacion_final + [individuos[i][0]]	
        return poblacion_final
                                     
    def agregar_poblacion(self, pob, hijos):
        poblacion = pob.copy()
        poblacion.extend(hijos)
        return poblacion

    def poda(self, pob, poblacion_maxima):
        poblacion=pob.copy()
        poblacion.sort(key=itemgetter(1,3))
        if poblacion.__len__() > poblacion_maxima:
            while poblacion.__len__() > poblacion_maxima:
                poblacion.remove(poblacion[-1])
        # Eliminar individuos repetidos
        for individuo in poblacion:
            while(poblacion.count(individuo) > 1):
                poblacion.remove(individuo)
        return poblacion  

    def ordenar_valores(self, valores):
        valores_ordenados = []
        valores_ordenar = []
        for i in range(valores.__len__()):
            valores_ordenar.append(valores[i][3])
        valores_ordenados = sorted(valores_ordenar, key = lambda x:[x])
        return valores_ordenados
        