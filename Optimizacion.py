import numpy as np

# =============================================================================
# 1. DATOS DE ENTRADA (CONFIGURACIÓN DEL PROBLEMA)
# =============================================================================
X_MIN, X_MAX = 0, 100
Y_MIN, Y_MAX = 0, 100
R_MAX = 20
N_MAX = 5        # Número máximo de torres
LAMBDA_SOL = 2.0 # Penalización por solapamiento
LAMBDA_COS = 0.1 # Penalización por costo operativo
DIM = 3 * N_MAX  # Dimensiones del problema (D)

def imprimir_datos_entrada():
    print("="*60)
    print("DATOS DE ENTRADA: CONFIGURACIÓN DEL PROBLEMA")
    print("="*60)
    print(f"-> Espacio de búsqueda : {X_MAX}x{Y_MAX} unidades")
    print(f"-> Número de torres    : {N_MAX}")
    print(f"-> Radio máximo/torre  : {R_MAX} unidades")
    print(f"-> Penalización (Solap): {LAMBDA_SOL}")
    print(f"-> Penalización (Costo): {LAMBDA_COS}")
    print(f"-> Dimensiones (D)     : {DIM}")
    print("="*60 + "\n")

# =============================================================================
# 2. FUNCIONES DE EVALUACIÓN Y TRADUCCIÓN (GENOTIPO -> FENOTIPO)
# =============================================================================
def limitar_espacio(individuo):
    """Limita la posición (x[i], I, S) dentro de las fronteras físicas."""
    ind = individuo.copy().reshape(N_MAX, 3)
    ind[:, 0] = np.clip(ind[:, 0], X_MIN, X_MAX)
    ind[:, 1] = np.clip(ind[:, 1], Y_MIN, Y_MAX)
    ind[:, 2] = np.clip(ind[:, 2], 0, R_MAX)
    return ind.flatten()

def funcion_objetivo(individuo):
    """Calcula la aptitud (Fitness) maximizando cobertura y minimizando penalizaciones."""
    torres = limitar_espacio(individuo).reshape(N_MAX, 3)
    grid = np.zeros((X_MAX, Y_MAX))
    xx, yy = np.meshgrid(np.arange(X_MAX), np.arange(Y_MAX))
    
    costo = 0
    for tx, ty, tr in torres:
        if tr <= 0: continue
        dist = np.sqrt((xx - tx)**2 + (yy - ty)**2)
        grid[dist <= tr] += 1
        costo += np.pi * (tr**2)
        
    cobertura = np.sum(grid >= 1)
    solapamiento = np.sum(grid > 1)
    return cobertura - (LAMBDA_SOL * solapamiento) - (LAMBDA_COS * costo)

def obtener_detalles_salida(individuo):
    """Decodifica el genotipo final para mostrar el desglose del fenotipo."""
    torres = limitar_espacio(individuo).reshape(N_MAX, 3)
    grid = np.zeros((X_MAX, Y_MAX))
    xx, yy = np.meshgrid(np.arange(X_MAX), np.arange(Y_MAX))
    
    costo = 0
    for tx, ty, tr in torres:
        if tr <= 0: continue
        dist = np.sqrt((xx - tx)**2 + (yy - ty)**2)
        grid[dist <= tr] += 1
        costo += np.pi * (tr**2)
        
    cobertura = np.sum(grid >= 1)
    solapamiento = np.sum(grid > 1)
    fitness = cobertura - (LAMBDA_SOL * solapamiento) - (LAMBDA_COS * costo)
    return torres, cobertura, solapamiento, costo, fitness

# =============================================================================
# 3. EL PROCESO: ALGORITMOS BIOINSPIRADOS (BASADOS EN TUS PSEUDOCÓDIGOS)
# =============================================================================

# --- PSO (Particle Swarm Optimization) ---
def ejecutar_pso(N=30, max_iter=100):
    print("--- Iniciando Proceso: PSO (Enjambre de Partículas) ---")
    w, c1, c2 = 0.7, 1.5, 1.5
    V_max = (X_MAX - X_MIN) * 0.1 # Límite de velocidad
    
    x = np.random.uniform([X_MIN, Y_MIN, 0]*N_MAX, [X_MAX, Y_MAX, R_MAX]*N_MAX, (N, DIM))
    v = np.random.uniform(-V_max, V_max, (N, DIM))
    
    pBest = x.copy()
    pBest_aptitud = np.array([funcion_objetivo(ind) for ind in x])
    
    gBest = pBest[np.argmax(pBest_aptitud)].copy()
    gBest_aptitud = np.max(pBest_aptitud)

    iteracion = 1
    while iteracion <= max_iter:
        for i in range(N):
            r1, r2 = np.random.rand(DIM), np.random.rand(DIM)
            
            # Ecuación de velocidad y control de movimiento
            v[i] = w*v[i] + c1*r1*(pBest[i]-x[i]) + c2*r2*(gBest-x[i])
            v[i] = np.clip(v[i], -V_max, V_max)
            x[i] = limitar_espacio(x[i] + v[i])
            
            aptitud_actual = funcion_objetivo(x[i])
            if aptitud_actual > pBest_aptitud[i]: 
                pBest[i] = x[i].copy()
                pBest_aptitud[i] = aptitud_actual
                
            if aptitud_actual > gBest_aptitud:
                gBest = x[i].copy()
                gBest_aptitud = aptitud_actual
                
        if iteracion == 1 or iteracion % 25 == 0:
            print(f"  Iteración {iteracion:03d} | Mejor Fitness: {gBest_aptitud:.2f}")
        iteracion += 1
    return gBest

# --- GWO (Grey Wolf Optimizer) ---
def ejecutar_gwo(N=30, max_iter=100):
    print("--- Iniciando Proceso: GWO (Optimizador Lobo Gris) ---")
    x = np.random.uniform([X_MIN, Y_MIN, 0]*N_MAX, [X_MAX, Y_MAX, R_MAX]*N_MAX, (N, DIM))
    alpha_pos, alpha_score = np.zeros(DIM), -np.inf
    beta_pos,  beta_score  = np.zeros(DIM), -np.inf
    delta_pos, delta_score = np.zeros(DIM), -np.inf

    iteracion = 1
    while iteracion <= max_iter:
        for i in range(N):
            x[i] = limitar_espacio(x[i])
            aptitud = funcion_objetivo(x[i])
            
            if aptitud > alpha_score:
                delta_pos, delta_score = beta_pos.copy(), beta_score
                beta_pos, beta_score = alpha_pos.copy(), alpha_score
                alpha_pos, alpha_score = x[i].copy(), aptitud
            elif aptitud > beta_score:
                delta_pos, delta_score = beta_pos.copy(), beta_score
                beta_pos, beta_score = x[i].copy(), aptitud
            elif aptitud > delta_score:
                delta_pos, delta_score = x[i].copy(), aptitud
        
        a = 2.0 - iteracion * (2.0 / max_iter)
        for i in range(N):
            r1, r2 = np.random.rand(DIM), np.random.rand(DIM)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            d_alpha = np.abs(C1 * alpha_pos - x[i])
            x1 = alpha_pos - A1 * d_alpha

            r1, r2 = np.random.rand(DIM), np.random.rand(DIM)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            d_beta = np.abs(C2 * beta_pos - x[i])
            x2 = beta_pos - A2 * d_beta

            r1, r2 = np.random.rand(DIM), np.random.rand(DIM)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            d_delta = np.abs(C3 * delta_pos - x[i])
            x3 = delta_pos - A3 * d_delta

            x[i] = (x1 + x2 + x3) / 3.0
            
        if iteracion == 1 or iteracion % 25 == 0:
            print(f"  Iteración {iteracion:03d} | Mejor Fitness (Alfa): {alpha_score:.2f}")
        iteracion += 1
    return alpha_pos

# --- AG (Algoritmo Genético) ---
def ejecutar_ag(N=30, max_gen=100):
    print("--- Iniciando Proceso: AG (Algoritmo Genético) ---")
    Pc = 0.8 # Probabilidad de cruce
    Pm = 0.1 # Probabilidad de mutación
    
    poblacion = np.random.uniform([X_MIN, Y_MIN, 0]*N_MAX, [X_MAX, Y_MAX, R_MAX]*N_MAX, (N, DIM))
    aptitudes = np.array([funcion_objetivo(ind) for ind in poblacion])
    mejor_historico = poblacion[np.argmax(aptitudes)].copy()
    
    gen = 1
    while gen <= max_gen:
        nueva_poblacion = []
        while len(nueva_poblacion) < N:
            # Selección por torneo
            t1, t2 = np.random.choice(N, 2, replace=False)
            padre1 = poblacion[t1] if aptitudes[t1] > aptitudes[t2] else poblacion[t2]
            t3, t4 = np.random.choice(N, 2, replace=False)
            padre2 = poblacion[t3] if aptitudes[t3] > aptitudes[t4] else poblacion[t4]
            
            # Cruce
            if np.random.rand() < Pc:
                punto = np.random.randint(1, DIM)
                hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
                hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
            else:
                hijo1, hijo2 = padre1.copy(), padre2.copy()
                
            # Mutación
            if np.random.rand() < Pm: hijo1 += np.random.normal(0, 2, DIM)
            if np.random.rand() < Pm: hijo2 += np.random.normal(0, 2, DIM)
            
            nueva_poblacion.append(limitar_espacio(hijo1))
            if len(nueva_poblacion) < N:
                nueva_poblacion.append(limitar_espacio(hijo2))
                
        poblacion = np.array(nueva_poblacion)
        aptitudes = np.array([funcion_objetivo(ind) for ind in poblacion])
        mejor_actual = poblacion[np.argmax(aptitudes)]
        
        if funcion_objetivo(mejor_actual) > funcion_objetivo(mejor_historico):
            mejor_historico = mejor_actual.copy()
            
        if gen == 1 or gen % 25 == 0:
            print(f"  Generación {gen:03d} | Mejor Fitness histórico: {funcion_objetivo(mejor_historico):.2f}")
        gen += 1
    return mejor_historico

# --- ABC (Artificial Bee Colony) ---
def ejecutar_abc(SN=30, max_iter=100):
    print("--- Iniciando Proceso: ABC (Colonia de Abejas) ---")
    limite = 20
    x = np.random.uniform([X_MIN, Y_MIN, 0]*N_MAX, [X_MAX, Y_MAX, R_MAX]*N_MAX, (SN, DIM))
    aptitud = np.array([funcion_objetivo(ind) for ind in x])
    intentos = np.zeros(SN)
    mejor_solucion = x[np.argmax(aptitud)].copy()

    iteracion = 1
    while iteracion <= max_iter:
        # Fase obreras
        for i in range(SN):
            k = np.random.choice([idx for idx in range(SN) if idx != i])
            j = np.random.randint(DIM)
            r = np.random.uniform(-1, 1)
            v_i = x[i].copy()
            v_i[j] = x[i, j] + r * (x[i, j] - x[k, j])
            v_i = limitar_espacio(v_i)
            
            aptitud_v = funcion_objetivo(v_i)
            if aptitud_v > aptitud[i]:
                x[i], aptitud[i], intentos[i] = v_i, aptitud_v, 0
            else:
                intentos[i] += 1
                
        # Calculo de probabilidades
        P = (aptitud - np.min(aptitud)) / (np.max(aptitud) - np.min(aptitud) + 1e-9)
        
        # Fase de selección (Observadoras)
        i = 0
        t = 0
        while t < SN:
            if np.random.rand() < P[i]:
                t += 1
                k = np.random.choice([idx for idx in range(SN) if idx != i])
                j = np.random.randint(DIM)
                r = np.random.uniform(-1, 1)
                v_i = x[i].copy()
                v_i[j] = x[i, j] + r * (x[i, j] - x[k, j])
                v_i = limitar_espacio(v_i)
                
                aptitud_v = funcion_objetivo(v_i)
                if aptitud_v > aptitud[i]:
                    x[i], aptitud[i], intentos[i] = v_i, aptitud_v, 0
                else:
                    intentos[i] += 1
            i = (i + 1) % SN
            
        mejor_actual = x[np.argmax(aptitud)]
        if funcion_objetivo(mejor_actual) > funcion_objetivo(mejor_solucion):
            mejor_solucion = mejor_actual.copy()
            
        # Fase exploradoras
        for i in range(SN):
            if intentos[i] >= limite:
                x[i] = np.random.uniform([X_MIN, Y_MIN, 0]*N_MAX, [X_MAX, Y_MAX, R_MAX]*N_MAX, DIM)
                aptitud[i] = funcion_objetivo(x[i])
                intentos[i] = 0
                
        if iteracion == 1 or iteracion % 25 == 0:
            print(f"  Iteración {iteracion:03d} | Mejor Fitness: {funcion_objetivo(mejor_solucion):.2f}")
        iteracion += 1
    return mejor_solucion

# --- AIS (Clonalg / Sistema Inmune) ---
def ejecutar_ais(N=30, max_iter=100):
    print("--- Iniciando Proceso: AIS (Sistema Inmune) ---")
    n_select = int(N * 0.5) 
    beta = 1.5              
    rho = 0.05              
    d = 5                   
    
    poblacion = np.random.uniform([X_MIN, Y_MIN, 0]*N_MAX, [X_MAX, Y_MAX, R_MAX]*N_MAX, (N, DIM))
    
    iteracion = 1
    while iteracion <= max_iter:
        afinidad = np.array([funcion_objetivo(ind) for ind in poblacion])
        idx_orden = np.argsort(afinidad)[::-1]
        poblacion_ordenada = poblacion[idx_orden]
        afinidad_ordenada = afinidad[idx_orden]
        
        poblacion_selecta = poblacion_ordenada[:n_select]
        afinidad_selecta = afinidad_ordenada[:n_select]
        poblacion_clones = []
        
        for i in range(n_select):
            anticuerpo_actual = poblacion_selecta[i]
            num_clones = int(np.round(beta * N / (i + 1)))
            afinidad_norm = (afinidad_selecta[i] - np.min(afinidad_selecta)) / (np.max(afinidad_selecta) - np.min(afinidad_selecta) + 1e-9)
            tasa_mutacion = np.exp(-rho * afinidad_norm)
            
            for _ in range(num_clones):
                clon = anticuerpo_actual.copy()
                clon_mutado = clon + np.random.normal(0, tasa_mutacion * 2, DIM)
                poblacion_clones.append(limitar_espacio(clon_mutado))
                
        afinidad_clones = np.array([funcion_objetivo(c) for c in poblacion_clones])
        poblacion_unida = np.vstack((poblacion_selecta, poblacion_clones))
        afinidad_unida = np.concatenate((afinidad_selecta, afinidad_clones))
        
        idx_unida = np.argsort(afinidad_unida)[::-1]
        poblacion_nueva = poblacion_unida[idx_unida][:N - d]
        
        nuevos_anticuerpos = np.random.uniform([X_MIN, Y_MIN, 0]*N_MAX, [X_MAX, Y_MAX, R_MAX]*N_MAX, (d, DIM))
        poblacion = np.vstack((poblacion_nueva, nuevos_anticuerpos))
        
        if iteracion == 1 or iteracion % 25 == 0:
            mejor_afinidad_iter = np.max([funcion_objetivo(ind) for ind in poblacion])
            print(f"  Iteración {iteracion:03d} | Mejor Afinidad: {mejor_afinidad_iter:.2f}")
        iteracion += 1
        
    afinidad_final = np.array([funcion_objetivo(ind) for ind in poblacion])
    idx_mejor = np.argsort(afinidad_final)[::-1][0]
    return poblacion[idx_mejor]

# =============================================================================
# 4. DATOS DE SALIDA (RESULTADOS FINALES)
# =============================================================================
def imprimir_resultados_algoritmo(nombre, individuo):
    torres, cobertura, solapamiento, costo, fitness = obtener_detalles_salida(individuo)
    print("\n" + "="*60)
    print(f"DATOS DE SALIDA: RESULTADOS FINALES PARA {nombre}")
    print("="*60)
    print(f"► FENOTIPO (Ubicación de Torres de Telecomunicación):")
    for i, (tx, ty, tr) in enumerate(torres):
        print(f"  Torre {i+1}: Coordenadas (X: {tx:5.1f}, Y: {ty:5.1f}) | Radio de alcance: {tr:4.1f}")
    
    print(f"\n► DESGLOSE DEL OBJETIVO:")
    print(f"  + Área total cubierta   : {cobertura:.1f} celdas")
    print(f"  - Costo operativo (área): {costo:.1f} unidades")
    print(f"  - Área solapada         : {solapamiento:.1f} celdas")
    print(f"  ----------------------------------------------------")
    print(f"  ★ FITNESS FINAL SCORE : {fitness:.2f}")
    print("="*60 + "\n")

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # 1. Imprimir Datos de Entrada
    imprimir_datos_entrada()
    
    algoritmos = {
        "PSO (Enjambre)": ejecutar_pso,
        "GWO (Lobos Grises)": ejecutar_gwo,
        "AG (Algoritmo Genético)": ejecutar_ag,
        "ABC (Colonia de Abejas)": ejecutar_abc,
        "AIS (Sistema Inmune)": ejecutar_ais
    }

    resultados = {}
    
    # 2. Ejecutar y mostrar el PROCESO de todos los algoritmos
    for nombre, func in algoritmos.items():
        mejor_ind = func()
        resultados[nombre] = mejor_ind
        print("") 
        
    # 3. Mostrar los DATOS DE SALIDA detallados de cada uno
    for nombre, individuo in resultados.items():
        imprimir_resultados_algoritmo(nombre, individuo)