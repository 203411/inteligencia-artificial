

class Neurona():
    def __init__(self, iteraciones, aprendizaje, umbral):
        self.iteraciones = iteraciones
        self.aprendizaje = aprendizaje
        self.umbral = umbral # Error permitido
        self.pesos = []
        self.X = []
        self.Y = []
    
    def leer_datos(self):
        df = pd.read_csv('datos2.csv')
        x = df.iloc[:, 0:5].values
        y = df.iloc[:, 5].values 
        x = self.bias(x)
        self.X = x
        self.Y = y
        self.pesos = np.random.rand(len(x[0]))
        
    def bias(self, x):
        x_bias = []
        for i in range(len(x)):
            x_bias.append([])
            x_bias[i].append(1)
            for j in range(len(x[i])):
                x_bias[i].append(x[i][j])
        return x_bias
    
    
    def calcular_u(self):
        w_transpuesta = np.transpose(self.pesos)
        u = (np.dot(self.X, w_transpuesta))
        return u
    
    def funcion_activacion(self, u):
        return u
     
    def delta_W(self, error):
        et = np.transpose(error)
        for i in range(len(self.pesos)):
            delta_W = (np.dot(et, self.X) * self.aprendizaje)
        return delta_W
    
    def nueva_w(self, delta_w):
        w_nuevo = self.pesos + delta_w
        self.pesos = w_nuevo
        return w_nuevo
    
    def calcular_error(self, yc):
        error = []
        for i in range(len(yc)):
            error.append(self.Y[i] - yc[i])
        return error
    
    def calcular_e(self, error): # Error en el conjunto de datos
        e = 0
        for i in range(len(error)):
            e += error[i]**2
        return e
    
    
def algoritmo(self):
    errores = []
    neurona.leer_datos()
    iteraciones = 0
    e = 100 # Inicializacion para poder entrar al while
    while e > neurona.umbral:
        u = neurona.calcular_u()
        yc = neurona.funcion_activacion(u)
        error = neurona.calcular_error(yc)
        delta_w = neurona.delta_W(error)
        neurona.nueva_w(delta_w)
        e = neurona.calcular_e(error)
        errores.append(e)
        iteraciones += 1
    return yc

# def iniciar_valores():
#     flag = True
#     try:
#         aprendizaje = float(interfaz.aprendizaje.text())
        
        