import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt

# Configurar matplotlib para evitar problemas de memoria
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

# ==================== PROBLEMA DE RED DE DATOS ====================

class RedDeDatos:
    """Define el problema de optimización de red"""
    
    def __init__(self):
        # Red de ejemplo: 5 nodos, 7 enlaces
        self.nodos = 5
        self.enlaces = [
            (0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4)
        ]
        self.num_enlaces = len(self.enlaces)
        self.dimension = self.num_enlaces * 2  # flujos + activaciones
        
        # Capacidades máximas por enlace (Mbps)
        self.capacidades = [100, 80, 120, 90, 110, 70, 100]
        
        # Latencia por enlace (ms)
        self.latencias = [5, 8, 3, 6, 4, 7, 5]
        
        # Costo por enlace (unidades monetarias)
        self.costos = [10, 12, 8, 15, 9, 11, 13]
        
        # Pesos de la función objetivo
        self.alpha = 0.5  # peso latencia
        self.beta = 0.3   # peso congestión
        self.gamma = 0.2  # peso costo
        
        # Demanda de tráfico entre nodos (fuente, destino, demanda)
        self.demandas = [
            (0, 4, 50),   # nodo0 a nodo4: 50 Mbps
            (0, 3, 30),   # nodo0 a nodo3: 30 Mbps
            (1, 4, 40),   # nodo1 a nodo4: 40 Mbps
            (2, 4, 35)    # nodo2 a nodo4: 35 Mbps
        ]
    
    def decodificar_solucion(self, x):
        """Convierte genotipo [0,1] a flujos y activaciones reales"""
        flujos = []
        activaciones = []
        
        # Primera mitad: flujos (normalizados a capacidades)
        for i in range(self.num_enlaces):
            flujo = x[i] * self.capacidades[i]
            flujos.append(flujo)
        
        # Segunda mitad: activaciones (umbral > 0.5)
        for i in range(self.num_enlaces):
            activacion = 1 if x[self.num_enlaces + i] > 0.5 else 0
            activaciones.append(activacion)
        
        return flujos, activaciones
    
    def verificar_flujo_conservacion(self, flujos):
        """Verifica ley de Kirchhoff (flujo que entra = flujo que sale)"""
        penalizacion = 0
        
        for demanda in self.demandas:
            origen, destino, demanda_mbps = demanda
            
            # Flujo neto en cada nodo para esta demanda
            for nodo in range(self.nodos):
                flujo_neto = 0
                for idx, (u, v) in enumerate(self.enlaces):
                    if flujos[idx] > 0:
                        if u == nodo:
                            flujo_neto -= flujos[idx]
                        elif v == nodo:
                            flujo_neto += flujos[idx]
                
                # Nodo origen debe enviar demanda
                if nodo == origen:
                    penalizacion += abs(flujo_neto + demanda_mbps)
                # Nodo destino debe recibir demanda
                elif nodo == destino:
                    penalizacion += abs(flujo_neto - demanda_mbps)
                # Nodos intermedios: flujo neto debe ser 0
                else:
                    penalizacion += abs(flujo_neto)
        
        return penalizacion
    
    def calcular_fitness(self, solucion):
        """Calcula la función objetivo (minimizar)"""
        flujos, activaciones = self.decodificar_solucion(solucion)
        
        # 1. Latencia total
        latencia_total = sum(flujos[i] * self.latencias[i] for i in range(self.num_enlaces))
        
        # 2. Congestión (violación de capacidad)
        congestion = 0
        for i in range(self.num_enlaces):
            if flujos[i] > self.capacidades[i]:
                congestion += (flujos[i] - self.capacidades[i]) ** 2
        
        # 3. Costo de enlaces activos
        costo_total = sum(activaciones[i] * self.costos[i] for i in range(self.num_enlaces))
        
        # 4. Penalización por violación de flujo
        penalizacion_flujo = self.verificar_flujo_conservacion(flujos)
        
        # Función objetivo ponderada
        fitness = (self.alpha * latencia_total + 
                   self.beta * congestion + 
                   self.gamma * costo_total +
                   10 * penalizacion_flujo)  # Penalización alta
        
        return fitness
    
    def limpiar_solucion(self, x):
        """Asegura que la solución esté en [0,1]"""
        return np.clip(x, 0, 1)

# ==================== ALGORITMO 1: PSO ====================

class PSO:
    """Particle Swarm Optimization"""
    
    def __init__(self, problema, N=30, max_iter=200, w=0.7, c1=1.5, c2=1.5):
        self.problema = problema
        self.N = N
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.D = problema.dimension
        
    def optimizar(self):
        D = self.D
        N = self.N
        
        # Inicialización
        x = np.random.rand(N, D)
        v = np.random.uniform(-0.1, 0.1, (N, D))
        
        pBest = x.copy()
        pBest_fitness = np.array([self.problema.calcular_fitness(x[i]) for i in range(N)])
        
        gBest_idx = np.argmin(pBest_fitness)
        gBest = pBest[gBest_idx].copy()
        gBest_fitness = pBest_fitness[gBest_idx]
        
        historial_fitness = []
        
        for iteracion in range(self.max_iter):
            for i in range(N):
                for d in range(D):
                    r1, r2 = np.random.rand(), np.random.rand()
                    v[i][d] = (self.w * v[i][d] + 
                               self.c1 * r1 * (pBest[i][d] - x[i][d]) +
                               self.c2 * r2 * (gBest[d] - x[i][d]))
                    v[i][d] = np.clip(v[i][d], -0.2, 0.2)
                    x[i][d] += v[i][d]
                    x[i][d] = np.clip(x[i][d], 0, 1)
                
                fitness = self.problema.calcular_fitness(x[i])
                
                if fitness < pBest_fitness[i]:
                    pBest[i] = x[i].copy()
                    pBest_fitness[i] = fitness
                    
                    if fitness < gBest_fitness:
                        gBest = x[i].copy()
                        gBest_fitness = fitness
            
            historial_fitness.append(gBest_fitness)
            
            if iteracion % 20 == 0:
                print(f"PSO - Iteracion {iteracion}: Mejor fitness = {gBest_fitness:.2f}")
        
        return gBest, gBest_fitness, historial_fitness

# ==================== ALGORITMO 2: GWO ====================

class GWO:
    """Grey Wolf Optimizer"""
    
    def __init__(self, problema, N=30, max_iter=200):
        self.problema = problema
        self.N = N
        self.max_iter = max_iter
        self.D = problema.dimension
        
    def optimizar(self):
        D = self.D
        N = self.N
        
        # Inicializar lobos
        wolves = np.random.rand(N, D)
        
        # Inicializar líderes (para minimización: score alto = malo)
        alpha_pos = np.zeros(D)
        beta_pos = np.zeros(D)
        delta_pos = np.zeros(D)
        
        alpha_score = float('inf')
        beta_score = float('inf')
        delta_score = float('inf')
        
        historial_fitness = []
        
        for iteracion in range(self.max_iter):
            # Evaluar todos los lobos
            for i in range(N):
                fitness = self.problema.calcular_fitness(wolves[i])
                
                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness
                    alpha_pos = wolves[i].copy()
                elif fitness < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness
                    beta_pos = wolves[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = wolves[i].copy()
            
            # Calcular parámetro 'a'
            a = 2.0 - iteracion * (2.0 / self.max_iter)
            
            # Actualizar posiciones
            for i in range(N):
                for d in range(D):
                    # Movimiento hacia alpha
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[d] - wolves[i][d])
                    X1 = alpha_pos[d] - A1 * D_alpha
                    
                    # Movimiento hacia beta
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[d] - wolves[i][d])
                    X2 = beta_pos[d] - A2 * D_beta
                    
                    # Movimiento hacia delta
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[d] - wolves[i][d])
                    X3 = delta_pos[d] - A3 * D_delta
                    
                    wolves[i][d] = (X1 + X2 + X3) / 3
                    wolves[i][d] = np.clip(wolves[i][d], 0, 1)
            
            historial_fitness.append(alpha_score)
            
            if iteracion % 20 == 0:
                print(f"GWO - Iteracion {iteracion}: Mejor fitness = {alpha_score:.2f}")
        
        return alpha_pos, alpha_score, historial_fitness

# ==================== ALGORITMO 3: AG ====================

class AlgoritmoGenetico:
    """Algoritmo Genetico"""
    
    def __init__(self, problema, N=50, max_gen=200, Pc=0.8, Pm=0.05):
        self.problema = problema
        self.N = N
        self.max_gen = max_gen
        self.Pc = Pc
        self.Pm = Pm
        self.D = problema.dimension
        
    def seleccion_torneo(self, poblacion, fitness, k=3):
        """Selección por torneo (minimización)"""
        mejor_idx = np.random.randint(len(poblacion))
        mejor_fitness = fitness[mejor_idx]
        
        for _ in range(k-1):
            idx = np.random.randint(len(poblacion))
            if fitness[idx] < mejor_fitness:
                mejor_fitness = fitness[idx]
                mejor_idx = idx
        
        return poblacion[mejor_idx].copy()
    
    def cruce_un_punto(self, padre1, padre2):
        """Cruce de un punto"""
        punto = np.random.randint(1, self.D)
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        return hijo1, hijo2
    
    def mutar(self, individuo):
        """Mutación por inserción de valor aleatorio"""
        for d in range(self.D):
            if np.random.rand() < self.Pm:
                individuo[d] = np.random.rand()
        return individuo
    
    def optimizar(self):
        # Inicialización
        poblacion = np.random.rand(self.N, self.D)
        fitness = np.array([self.problema.calcular_fitness(ind) for ind in poblacion])
        
        mejor_idx = np.argmin(fitness)
        mejor_historico = poblacion[mejor_idx].copy()
        mejor_fitness_historico = fitness[mejor_idx]
        
        historial_fitness = []
        
        for generacion in range(self.max_gen):
            nueva_poblacion = []
            
            while len(nueva_poblacion) < self.N:
                # Selección
                padre1 = self.seleccion_torneo(poblacion, fitness)
                padre2 = self.seleccion_torneo(poblacion, fitness)
                
                # Cruce
                if np.random.rand() < self.Pc:
                    hijo1, hijo2 = self.cruce_un_punto(padre1, padre2)
                else:
                    hijo1, hijo2 = padre1, padre2
                
                # Mutación
                hijo1 = self.mutar(hijo1)
                hijo2 = self.mutar(hijo2)
                
                nueva_poblacion.append(hijo1)
                if len(nueva_poblacion) < self.N:
                    nueva_poblacion.append(hijo2)
            
            poblacion = np.array(nueva_poblacion)
            fitness = np.array([self.problema.calcular_fitness(ind) for ind in poblacion])
            
            mejor_actual_idx = np.argmin(fitness)
            if fitness[mejor_actual_idx] < mejor_fitness_historico:
                mejor_historico = poblacion[mejor_actual_idx].copy()
                mejor_fitness_historico = fitness[mejor_actual_idx]
            
            historial_fitness.append(mejor_fitness_historico)
            
            if generacion % 20 == 0:
                print(f"AG - Generacion {generacion}: Mejor fitness = {mejor_fitness_historico:.2f}")
        
        return mejor_historico, mejor_fitness_historico, historial_fitness

# ==================== ALGORITMO 4: ABC ====================

class ABC:
    """Artificial Bee Colony"""
    
    def __init__(self, problema, SN=30, max_iter=200, limite=20):
        self.problema = problema
        self.SN = SN
        self.max_iter = max_iter
        self.limite = limite
        self.D = problema.dimension
        
    def calcular_aptitud(self, fitness):
        """Convierte fitness (minimización) a aptitud (maximización)"""
        return 1.0 / (1.0 + fitness)
    
    def optimizar(self):
        D = self.D
        SN = self.SN
        
        # Inicialización
        fuentes = np.random.rand(SN, D)
        fitness = np.array([self.problema.calcular_fitness(f) for f in fuentes])
        aptitud = np.array([self.calcular_aptitud(f) for f in fitness])
        intentos = np.zeros(SN, dtype=int)
        
        mejor_idx = np.argmin(fitness)
        mejor_solucion = fuentes[mejor_idx].copy()
        mejor_fitness = fitness[mejor_idx]
        
        historial_fitness = []
        
        for iteracion in range(self.max_iter):
            # Fase de abejas obreras
            for i in range(SN):
                # Seleccionar vecino
                k = i
                while k == i:
                    k = np.random.randint(SN)
                
                # Seleccionar dimensión
                j = np.random.randint(D)
                
                # Mutación
                r = np.random.uniform(-1, 1)
                nueva_fuente = fuentes[i].copy()
                nueva_fuente[j] = fuentes[i][j] + r * (fuentes[i][j] - fuentes[k][j])
                nueva_fuente = np.clip(nueva_fuente, 0, 1)
                
                # Evaluar
                nuevo_fitness = self.problema.calcular_fitness(nueva_fuente)
                nuevo_aptitud = self.calcular_aptitud(nuevo_fitness)
                
                # Selección voraz
                if nuevo_aptitud > aptitud[i]:
                    fuentes[i] = nueva_fuente
                    fitness[i] = nuevo_fitness
                    aptitud[i] = nuevo_aptitud
                    intentos[i] = 0
                else:
                    intentos[i] += 1
            
            # Calcular probabilidades para abejas observadoras
            probabilidades = aptitud / np.sum(aptitud)
            
            # Fase de abejas observadoras
            t = 0
            i = 0
            while t < SN:
                if np.random.rand() < probabilidades[i]:
                    t += 1
                    
                    # Misma mutación que en obreras
                    k = i
                    while k == i:
                        k = np.random.randint(SN)
                    
                    j = np.random.randint(D)
                    r = np.random.uniform(-1, 1)
                    nueva_fuente = fuentes[i].copy()
                    nueva_fuente[j] = fuentes[i][j] + r * (fuentes[i][j] - fuentes[k][j])
                    nueva_fuente = np.clip(nueva_fuente, 0, 1)
                    
                    nuevo_fitness = self.problema.calcular_fitness(nueva_fuente)
                    nuevo_aptitud = self.calcular_aptitud(nuevo_fitness)
                    
                    if nuevo_aptitud > aptitud[i]:
                        fuentes[i] = nueva_fuente
                        fitness[i] = nuevo_fitness
                        aptitud[i] = nuevo_aptitud
                        intentos[i] = 0
                    else:
                        intentos[i] += 1
                
                i = (i + 1) % SN
            
            # Actualizar mejor solución
            mejor_idx_actual = np.argmin(fitness)
            if fitness[mejor_idx_actual] < mejor_fitness:
                mejor_fitness = fitness[mejor_idx_actual]
                mejor_solucion = fuentes[mejor_idx_actual].copy()
            
            historial_fitness.append(mejor_fitness)
            
            # Fase de abejas exploradoras
            for i in range(SN):
                if intentos[i] >= self.limite:
                    fuentes[i] = np.random.rand(D)
                    fitness[i] = self.problema.calcular_fitness(fuentes[i])
                    aptitud[i] = self.calcular_aptitud(fitness[i])
                    intentos[i] = 0
            
            if iteracion % 20 == 0:
                print(f"ABC - Iteracion {iteracion}: Mejor fitness = {mejor_fitness:.2f}")
        
        return mejor_solucion, mejor_fitness, historial_fitness

# ==================== ALGORITMO 5: AIS (Clonalg) ====================

class AIS:
    """Artificial Immune System - Clonalg"""
    
    def __init__(self, problema, N=40, n_select=10, beta=1, rho=0.1, d=5, max_iter=200):
        self.problema = problema
        self.N = N
        self.n_select = n_select
        self.beta = beta
        self.rho = rho
        self.d = d
        self.max_iter = max_iter
        self.D = problema.dimension
        
    def calcular_afinidad(self, fitness):
        """Convierte fitness a afinidad (mayor = mejor)"""
        return 1.0 / (1.0 + fitness)
    
    def mutar_clon(self, anticuerpo, tasa_mutacion):
        """Hipermutación proporcional a la tasa"""
        clon = anticuerpo.copy()
        for d in range(self.D):
            if np.random.rand() < tasa_mutacion:
                clon[d] = np.random.rand()
        return clon
    
    def optimizar(self):
        D = self.D
        N = self.N
        
        # Inicialización
        poblacion = np.random.rand(N, D)
        fitness = np.array([self.problema.calcular_fitness(ind) for ind in poblacion])
        afinidad = np.array([self.calcular_afinidad(f) for f in fitness])
        
        mejor_idx = np.argmin(fitness)
        mejor_solucion = poblacion[mejor_idx].copy()
        mejor_fitness = fitness[mejor_idx]
        
        historial_fitness = []
        
        for iteracion in range(self.max_iter):
            # Ordenar por afinidad descendente (mejores primero)
            idx_orden = np.argsort(afinidad)[::-1]
            poblacion = poblacion[idx_orden]
            afinidad = afinidad[idx_orden]
            fitness = fitness[idx_orden]
            
            # Seleccionar los mejores
            poblacion_selecta = poblacion[:self.n_select]
            afinidad_selecta = afinidad[:self.n_select]
            
            # Clonación y mutación
            clones = []
            for i in range(self.n_select):
                # Número de clones (inversamente proporcional al rango)
                num_clones = int(round(self.beta * N / (i + 1)))
                num_clones = max(1, min(num_clones, N))
                
                # Tasa de mutación (inversamente proporcional a la afinidad)
                tasa_mutacion = np.exp(-self.rho * afinidad_selecta[i])
                
                for _ in range(num_clones):
                    clon = self.mutar_clon(poblacion_selecta[i], tasa_mutacion)
                    clones.append(clon)
            
            clones = np.array(clones)
            
            # Evaluar clones
            fitness_clones = np.array([self.problema.calcular_fitness(clon) for clon in clones])
            afinidad_clones = np.array([self.calcular_afinidad(f) for f in fitness_clones])
            
            # Unir población selecta con clones
            union_poblacion = np.vstack([poblacion_selecta, clones])
            union_afinidad = np.concatenate([afinidad_selecta, afinidad_clones])
            union_fitness = np.concatenate([fitness[:self.n_select], fitness_clones])
            
            # Ordenar por afinidad
            idx_orden = np.argsort(union_afinidad)[::-1]
            union_poblacion = union_poblacion[idx_orden]
            union_afinidad = union_afinidad[idx_orden]
            union_fitness = union_fitness[idx_orden]
            
            # Seleccionar los N-d mejores
            nueva_poblacion = union_poblacion[:N - self.d].copy()
            nuevo_fitness = union_fitness[:N - self.d].copy()
            
            # Insertar nuevos anticuerpos aleatorios (diversidad)
            nuevos = np.random.rand(self.d, D)
            nuevos_fitness = np.array([self.problema.calcular_fitness(n) for n in nuevos])
            
            poblacion = np.vstack([nueva_poblacion, nuevos])
            fitness = np.concatenate([nuevo_fitness, nuevos_fitness])
            afinidad = np.array([self.calcular_afinidad(f) for f in fitness])
            
            # Actualizar mejor solución
            mejor_actual_idx = np.argmin(fitness)
            if fitness[mejor_actual_idx] < mejor_fitness:
                mejor_fitness = fitness[mejor_actual_idx]
                mejor_solucion = poblacion[mejor_actual_idx].copy()
            
            historial_fitness.append(mejor_fitness)
            
            if iteracion % 20 == 0:
                print(f"AIS - Iteracion {iteracion}: Mejor fitness = {mejor_fitness:.2f}")
        
        return mejor_solucion, mejor_fitness, historial_fitness

# ==================== FUNCIONES DE VISUALIZACION ====================

def grafica_convergencia(resultados):
    """Grafica 1: Convergencia de todos los algoritmos"""
    plt.figure(figsize=(10, 6))
    
    colores = {'PSO': 'blue', 'GWO': 'green', 'AG': 'red', 'ABC': 'orange', 'AIS': 'purple'}
    estilos = {'PSO': '-', 'GWO': '-', 'AG': '--', 'ABC': '-.', 'AIS': ':'}
    
    for nombre, datos in resultados.items():
        plt.plot(datos['historial'], 
                label=nombre, 
                color=colores[nombre],
                linestyle=estilos[nombre],
                linewidth=1.5)
    
    plt.xlabel('Iteracion / Generacion', fontsize=11)
    plt.ylabel('Fitness (Latencia + Congestion + Costo)', fontsize=11)
    plt.title('Figura 1: Convergencia de Algoritmos Bioinspirados', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('1_convergencia_algoritmos.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  - Grafica 1 guardada: 1_convergencia_algoritmos.png")

def grafica_caja_resultados(resultados):
    """Grafica 2: Diagrama de caja de los ultimos 50 fitness"""
    plt.figure(figsize=(9, 5))
    
    datos_caja = []
    etiquetas = []
    
    for nombre, datos in resultados.items():
        # Tomar los últimos 50 valores o todos si hay menos
        ultimos = datos['historial'][-50:] if len(datos['historial']) >= 50 else datos['historial']
        datos_caja.append(ultimos)
        etiquetas.append(nombre)
    
    # CORREGIDO: labels -> tick_labels (soluciona el warning)
    bp = plt.boxplot(datos_caja, tick_labels=etiquetas, patch_artist=True)
    
    # Colorear las cajas
    colores_caja = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'plum']
    for patch, color in zip(bp['boxes'], colores_caja):
        patch.set_facecolor(color)
    
    plt.xlabel('Algoritmo', fontsize=11)
    plt.ylabel('Fitness', fontsize=11)
    plt.title('Figura 2: Distribucion de Fitness en las Ultimas Iteraciones', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('2_caja_resultados.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  - Grafica 2 guardada: 2_caja_resultados.png")

def grafica_mejor_fitness(resultados):
    """Grafica 3: Mejor fitness alcanzado por cada algoritmo"""
    plt.figure(figsize=(9, 5))
    
    nombres = list(resultados.keys())
    mejores = [resultados[n]['fitness'] for n in nombres]
    
    colores_barra = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    barras = plt.bar(nombres, mejores, color=colores_barra, edgecolor='black', linewidth=1)
    
    # Agregar valores en las barras
    for barra, valor in zip(barras, mejores):
        plt.text(barra.get_x() + barra.get_width()/2, 
                barra.get_height() + 5, 
                f'{valor:.2f}', 
                ha='center', 
                va='bottom',
                fontsize=9)
    
    plt.xlabel('Algoritmo', fontsize=11)
    plt.ylabel('Mejor Fitness Alcanzado', fontsize=11)
    plt.title('Figura 3: Comparacion del Mejor Fitness por Algoritmo', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('3_mejor_fitness.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  - Grafica 3 guardada: 3_mejor_fitness.png")

def grafica_mejora_relativa(resultados):
    """Grafica 4: Mejora porcentual respecto al peor algoritmo"""
    plt.figure(figsize=(9, 5))
    
    # Encontrar el peor fitness
    peor_fitness = max([resultados[n]['fitness'] for n in resultados.keys()])
    
    nombres = list(resultados.keys())
    mejora = [(peor_fitness - resultados[n]['fitness']) / peor_fitness * 100 for n in nombres]
    
    colores = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    barras = plt.barh(nombres, mejora, color=colores, edgecolor='black', linewidth=1)
    
    # Agregar valores
    for barra, valor in zip(barras, mejora):
        plt.text(barra.get_width() + 0.2, 
                barra.get_y() + barra.get_height()/2, 
                f'{valor:.1f}%', 
                ha='left', 
                va='center',
                fontsize=9)
    
    plt.xlabel('Mejora Porcentual (%)', fontsize=11)
    plt.ylabel('Algoritmo', fontsize=11)
    plt.title('Figura 4: Mejora Relativa Respecto al Peor Algoritmo', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')
    plt.tight_layout()
    plt.savefig('4_mejora_relativa.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  - Grafica 4 guardada: 4_mejora_relativa.png")

def grafica_red(problema, mejor_solucion, mejor_nombre):
    """Grafica 5: Topologia de la red optimizada"""
    plt.figure(figsize=(9, 7))
    
    flujos, activaciones = problema.decodificar_solucion(mejor_solucion)
    
    # Posiciones de los nodos
    posiciones = {
        0: (0, 2),
        1: (2, 3),
        2: (2, 1.5),
        3: (4, 2.5),
        4: (4, 0.5)
    }
    
    # Dibujar nodos
    for nodo, (x, y) in posiciones.items():
        circulo = plt.Circle((x, y), 0.2, color='lightblue', ec='black', linewidth=2, zorder=2)
        plt.gca().add_patch(circulo)
        plt.text(x, y, str(nodo), ha='center', va='center', fontweight='bold', fontsize=12)
    
    # Dibujar enlaces
    for idx, (u, v) in enumerate(problema.enlaces):
        x1, y1 = posiciones[u]
        x2, y2 = posiciones[v]
        
        if activaciones[idx] == 1:
            color = 'green'
            width = 1.5 + (flujos[idx] / max(problema.capacidades)) * 4
            alpha = 0.8
            # Etiqueta con flujo solo para enlaces activos
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            plt.text(mx, my, f'{flujos[idx]:.0f}', 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        else:
            color = 'red'
            width = 1
            alpha = 0.3
        
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha, zorder=1)
    
    plt.xlim(-0.5, 5)
    plt.ylim(0, 3.5)
    plt.axis('off')
    plt.title(f'Figura 5: Topologia Optimizada de la Red (Mejor: {mejor_nombre})\nVerde=Activo, Rojo=Inactivo', 
             fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('5_topologia_red.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  - Grafica 5 guardada: 5_topologia_red.png")

# ==================== EJECUCION Y COMPARACION ====================

def main():
    print("="*60)
    print("TAREA 4 - OPTIMIZACION DE RED DE DATOS")
    print("="*60)
    
    # Crear problema
    problema = RedDeDatos()
    
    print(f"\nProblema: Red con {problema.nodos} nodos y {problema.num_enlaces} enlaces")
    print(f"Dimensiones del problema: {problema.dimension}")
    
    # Ejecutar todos los algoritmos
    algoritmos = {
        "PSO": PSO(problema, N=30, max_iter=200),
        "GWO": GWO(problema, N=30, max_iter=200),
        "AG": AlgoritmoGenetico(problema, N=50, max_gen=200),
        "ABC": ABC(problema, SN=30, max_iter=200),
        "AIS": AIS(problema, N=40, max_iter=200)
    }
    
    resultados = {}
    
    for nombre, algoritmo in algoritmos.items():
        print("\n" + "="*60)
        print(f"Ejecutando {nombre}...")
        print("="*60)
        
        mejor_sol, mejor_fit, historial = algoritmo.optimizar()
        resultados[nombre] = {
            'fitness': mejor_fit,
            'historial': historial,
            'solucion': mejor_sol
        }
        
        print(f"\nOK {nombre} - Mejor fitness encontrado: {mejor_fit:.2f}")
    
    # Mostrar resultados comparativos
    print("\n" + "="*60)
    print("RESULTADOS COMPARATIVOS")
    print("="*60)
    
    print("\n| Algoritmo | Mejor Fitness |")
    print("|-----------|---------------|")
    for nombre, datos in sorted(resultados.items(), key=lambda x: x[1]['fitness']):
        print(f"| {nombre:9} | {datos['fitness']:12.2f} |")
    
    # Generar todas las graficas
    print("\n" + "="*60)
    print("GENERANDO GRAFICAS...")
    print("="*60)
    
    grafica_convergencia(resultados)
    grafica_caja_resultados(resultados)
    grafica_mejor_fitness(resultados)
    grafica_mejora_relativa(resultados)
    
    # Mejor algoritmo
    mejor_algoritmo = min(resultados.items(), key=lambda x: x[1]['fitness'])
    grafica_red(problema, mejor_algoritmo[1]['solucion'], mejor_algoritmo[0])
    
    # Analisis final
    print("\n" + "="*60)
    print(f"MEJOR ALGORITMO: {mejor_algoritmo[0]}")
    print(f"Fitness alcanzado: {mejor_algoritmo[1]['fitness']:.2f}")
    print("="*60)
    
    # Decodificar la mejor solución
    flujos, activaciones = problema.decodificar_solucion(mejor_algoritmo[1]['solucion'])
    
    print("\nConfiguracion optima de la red:")
    print("-" * 60)
    for i, (enlace, flujo) in enumerate(zip(problema.enlaces, flujos)):
        estado = "ACTIVO" if activaciones[i] == 1 else "INACTIVO"
        capacidad = problema.capacidades[i]
        porcentaje = (flujo / capacidad) * 100 if capacidad > 0 else 0
        print(f"Enlace {enlace[0]}-{enlace[1]}: {flujo:.1f} / {capacidad} Mbps ({porcentaje:.1f}%) - {estado}")
    
    # Calcular estadisticas
    print("\nESTADISTICAS DE LOS ALGORITMOS:")
    print("-" * 60)
    for nombre, datos in resultados.items():
        historial = datos['historial']
        print(f"{nombre:4} - Mejor: {datos['fitness']:8.2f} | "
              f"Promedio ultimas 20: {np.mean(historial[-20:]):8.2f} | "
              f"Desv. estandar: {np.std(historial[-20:]):6.2f}")
    
    print("\n" + "="*60)
    print("TAREA COMPLETADA - SE GENERARON 5 GRAFICAS")
    print("Archivos guardados:")
    print("   - 1_convergencia_algoritmos.png")
    print("   - 2_caja_resultados.png")
    print("   - 3_mejor_fitness.png")
    print("   - 4_mejora_relativa.png")
    print("   - 5_topologia_red.png")
    print("="*60)

if __name__ == "__main__":
    main()