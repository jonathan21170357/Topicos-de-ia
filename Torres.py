import numpy as np
import random
import matplotlib.pyplot as plt

# ==================== CONFIGURACIÓN DEL PROBLEMA ====================
N_TORRES = 4
ANCHO = 100
ALTO = 100
RADIO = 25
N_PUNTOS_DEMANDA = 50
POBLACION = 30
MAX_ITER = 100

# ==================== FUNCIÓN OBJETIVO ====================
def evaluar_solucion(x, n_torres, ancho, alto, radio, puntos_demanda, demanda):
    torres = x.reshape(n_torres, 2)
    
    # Penalización fuera de límites
    penalizacion = 0
    for t in torres:
        if t[0] < 0 or t[0] > ancho or t[1] < 0 or t[1] > alto:
            penalizacion += 1000
    
    # Cobertura
    cobertura = 0
    for punto, dem in zip(puntos_demanda, demanda):
        for torre in torres:
            if np.linalg.norm(punto - torre) <= radio:
                cobertura += dem
                break
    
    # Interferencia (torres muy juntas)
    interferencia = 0
    for i in range(n_torres):
        for j in range(i+1, n_torres):
            dist = np.linalg.norm(torres[i] - torres[j])
            if dist < radio * 0.7:
                interferencia += (radio * 0.7 - dist) * 10
    
    return cobertura - interferencia - penalizacion

# ==================== FUNCIONES AUXILIARES ====================
def generar_solucion_aleatoria(D, limites):
    return np.array([random.uniform(limites[d][0], limites[d][1]) for d in range(D)])

def limitar_solucion(x, limites):
    return np.clip(x, [l[0] for l in limites], [l[1] for l in limites])

# ==================== ALGORITMO PSO ====================
def pso(problema_params):
    n_torres, ancho, alto, radio, puntos_demanda, demanda, D, limites = problema_params
    N, max_iter = POBLACION, MAX_ITER
    w, c1, c2 = 0.7, 1.5, 1.5
    
    v_max = np.array([(limites[d][1] - limites[d][0]) * 0.2 for d in range(D)])
    
    # Inicialización
    posiciones = np.array([generar_solucion_aleatoria(D, limites) for _ in range(N)])
    velocidades = np.random.uniform(-v_max, v_max, (N, D))
    pbest = posiciones.copy()
    pbest_fit = np.array([evaluar_solucion(p, n_torres, ancho, alto, radio, puntos_demanda, demanda) for p in posiciones])
    gbest = pbest[np.argmax(pbest_fit)].copy()
    gbest_fit = max(pbest_fit)
    historial = []
    
    for iteracion in range(max_iter):
        for i in range(N):
            r1, r2 = random.random(), random.random()
            velocidades[i] = w * velocidades[i] + c1 * r1 * (pbest[i] - posiciones[i]) + c2 * r2 * (gbest - posiciones[i])
            velocidades[i] = np.clip(velocidades[i], -v_max, v_max)
            posiciones[i] = limitar_solucion(posiciones[i] + velocidades[i], limites)
            
            fit = evaluar_solucion(posiciones[i], n_torres, ancho, alto, radio, puntos_demanda, demanda)
            if fit > pbest_fit[i]:
                pbest[i], pbest_fit[i] = posiciones[i].copy(), fit
                if fit > gbest_fit:
                    gbest, gbest_fit = posiciones[i].copy(), fit
        
        historial.append(gbest_fit)
        if (iteracion + 1) % 20 == 0:
            print(f"PSO - Iteración {iteracion+1}: {gbest_fit:.2f}")
    
    return gbest, gbest_fit, historial

# ==================== ALGORITMO GWO ====================
def gwo(problema_params):
    n_torres, ancho, alto, radio, puntos_demanda, demanda, D, limites = problema_params
    N, max_iter = POBLACION, MAX_ITER
    
    lobos = np.array([generar_solucion_aleatoria(D, limites) for _ in range(N)])
    aptitudes = np.array([evaluar_solucion(l, n_torres, ancho, alto, radio, puntos_demanda, demanda) for l in lobos])
    
    idx = np.argsort(aptitudes)[::-1]
    alpha_pos, alpha_score = lobos[idx[0]].copy(), aptitudes[idx[0]]
    beta_pos, beta_score = lobos[idx[1]].copy(), aptitudes[idx[1]]
    delta_pos, delta_score = lobos[idx[2]].copy(), aptitudes[idx[2]]
    historial = []
    
    for iteracion in range(max_iter):
        a = 2.0 - iteracion * (2.0 / max_iter)
        
        for i in range(N):
            for d in range(D):
                r1, r2 = random.random(), random.random()
                A1, C1 = 2*a*r1 - a, 2*r2
                D_alpha = abs(C1 * alpha_pos[d] - lobos[i][d])
                x1 = alpha_pos[d] - A1 * D_alpha
                
                r1, r2 = random.random(), random.random()
                A2, C2 = 2*a*r1 - a, 2*r2
                D_beta = abs(C2 * beta_pos[d] - lobos[i][d])
                x2 = beta_pos[d] - A2 * D_beta
                
                r1, r2 = random.random(), random.random()
                A3, C3 = 2*a*r1 - a, 2*r2
                D_delta = abs(C3 * delta_pos[d] - lobos[i][d])
                x3 = delta_pos[d] - A3 * D_delta
                
                lobos[i][d] = (x1 + x2 + x3) / 3
            
            lobos[i] = limitar_solucion(lobos[i], limites)
            aptitud = evaluar_solucion(lobos[i], n_torres, ancho, alto, radio, puntos_demanda, demanda)
            
            if aptitud > alpha_score:
                delta_pos, delta_score = beta_pos.copy(), beta_score
                beta_pos, beta_score = alpha_pos.copy(), alpha_score
                alpha_pos, alpha_score = lobos[i].copy(), aptitud
            elif aptitud > beta_score:
                delta_pos, delta_score = beta_pos.copy(), beta_score
                beta_pos, beta_score = lobos[i].copy(), aptitud
            elif aptitud > delta_score:
                delta_pos, delta_score = lobos[i].copy(), aptitud
        
        historial.append(alpha_score)
        if (iteracion + 1) % 20 == 0:
            print(f"GWO - Iteración {iteracion+1}: {alpha_score:.2f}")
    
    return alpha_pos, alpha_score, historial

# ==================== ALGORITMO GENÉTICO ====================
def ag(problema_params):
    n_torres, ancho, alto, radio, puntos_demanda, demanda, D, limites = problema_params
    N, max_gen = POBLACION, MAX_ITER
    pc, pm = 0.8, 0.1
    
    poblacion = np.array([generar_solucion_aleatoria(D, limites) for _ in range(N)])
    aptitudes = np.array([evaluar_solucion(ind, n_torres, ancho, alto, radio, puntos_demanda, demanda) for ind in poblacion])
    mejor_historico = poblacion[np.argmax(aptitudes)].copy()
    mejor_fit = max(aptitudes)
    historial = []
    
    def cruzar(p1, p2):
        if random.random() < pc:
            punto = random.randint(1, D-1)
            return np.concatenate([p1[:punto], p2[punto:]]), np.concatenate([p2[:punto], p1[punto:]])
        return p1.copy(), p2.copy()
    
    def mutar(x):
        for d in range(D):
            if random.random() < pm:
                rango = limites[d][1] - limites[d][0]
                x[d] += random.gauss(0, rango * 0.1)
        return limitar_solucion(x, limites)
    
    def seleccion_torneo(k=3):
        idx = random.sample(range(N), k)
        return poblacion[idx[np.argmax(aptitudes[idx])]].copy()
    
    for generacion in range(max_gen):
        nueva_poblacion = []
        while len(nueva_poblacion) < N:
            padre1, padre2 = seleccion_torneo(), seleccion_torneo()
            hijo1, hijo2 = cruzar(padre1, padre2)
            nueva_poblacion.append(mutar(hijo1))
            if len(nueva_poblacion) < N:
                nueva_poblacion.append(mutar(hijo2))
        
        poblacion = np.array(nueva_poblacion)
        aptitudes = np.array([evaluar_solucion(ind, n_torres, ancho, alto, radio, puntos_demanda, demanda) for ind in poblacion])
        
        mejor_actual_idx = np.argmax(aptitudes)
        if aptitudes[mejor_actual_idx] > mejor_fit:
            mejor_historico = poblacion[mejor_actual_idx].copy()
            mejor_fit = aptitudes[mejor_actual_idx]
        
        historial.append(mejor_fit)
        if (generacion + 1) % 20 == 0:
            print(f"AG - Generación {generacion+1}: {mejor_fit:.2f}")
    
    return mejor_historico, mejor_fit, historial

# ==================== ALGORITMO ABC ====================
def abc(problema_params):
    n_torres, ancho, alto, radio, puntos_demanda, demanda, D, limites = problema_params
    SN, max_iter = POBLACION, MAX_ITER
    limite_fracaso = 20
    
    fuentes = np.array([generar_solucion_aleatoria(D, limites) for _ in range(SN)])
    aptitudes = np.array([evaluar_solucion(f, n_torres, ancho, alto, radio, puntos_demanda, demanda) for f in fuentes])
    intentos = np.zeros(SN)
    mejor_solucion = fuentes[np.argmax(aptitudes)].copy()
    mejor_fit = max(aptitudes)
    historial = []
    
    for iteracion in range(max_iter):
        # Fase abejas obreras
        for i in range(SN):
            k = random.choice([x for x in range(SN) if x != i])
            j = random.randint(0, D-1)
            r = random.uniform(-1, 1)
            v = fuentes[i].copy()
            v[j] = fuentes[i][j] + r * (fuentes[i][j] - fuentes[k][j])
            v = limitar_solucion(v, limites)
            v_fit = evaluar_solucion(v, n_torres, ancho, alto, radio, puntos_demanda, demanda)
            
            if v_fit > aptitudes[i]:
                fuentes[i], aptitudes[i], intentos[i] = v, v_fit, 0
            else:
                intentos[i] += 1
        
        # Calcular probabilidades
        probabilidades = aptitudes / (aptitudes.sum() + 1e-10)
        
        # Fase abejas observadoras
        i, t = 0, 0
        while t < SN:
            if random.random() < probabilidades[i]:
                t += 1
                k = random.choice([x for x in range(SN) if x != i])
                j = random.randint(0, D-1)
                r = random.uniform(-1, 1)
                v = fuentes[i].copy()
                v[j] = fuentes[i][j] + r * (fuentes[i][j] - fuentes[k][j])
                v = limitar_solucion(v, limites)
                v_fit = evaluar_solucion(v, n_torres, ancho, alto, radio, puntos_demanda, demanda)
                
                if v_fit > aptitudes[i]:
                    fuentes[i], aptitudes[i], intentos[i] = v, v_fit, 0
                else:
                    intentos[i] += 1
            i = (i + 1) % SN
        
        # Fase abejas exploradoras
        for i in range(SN):
            if intentos[i] >= limite_fracaso:
                fuentes[i] = generar_solucion_aleatoria(D, limites)
                aptitudes[i] = evaluar_solucion(fuentes[i], n_torres, ancho, alto, radio, puntos_demanda, demanda)
                intentos[i] = 0
        
        mejor_idx = np.argmax(aptitudes)
        if aptitudes[mejor_idx] > mejor_fit:
            mejor_solucion, mejor_fit = fuentes[mejor_idx].copy(), aptitudes[mejor_idx]
        
        historial.append(mejor_fit)
        if (iteracion + 1) % 20 == 0:
            print(f"ABC - Iteración {iteracion+1}: {mejor_fit:.2f}")
    
    return mejor_solucion, mejor_fit, historial

# ==================== ALGORITMO AIS ====================
def ais(problema_params):
    n_torres, ancho, alto, radio, puntos_demanda, demanda, D, limites = problema_params
    N, max_iter = POBLACION, MAX_ITER
    n_select = N // 3
    beta, rho, d = 5.0, 1.0, 5
    
    poblacion = np.array([generar_solucion_aleatoria(D, limites) for _ in range(N)])
    afinidades = np.array([evaluar_solucion(p, n_torres, ancho, alto, radio, puntos_demanda, demanda) for p in poblacion])
    mejor_anticuerpo = poblacion[np.argmax(afinidades)].copy()
    mejor_afinidad = max(afinidades)
    historial = []
    
    def mutar(anticuerpo, tasa):
        mutado = anticuerpo.copy()
        for d in range(D):
            if random.random() < tasa:
                rango = limites[d][1] - limites[d][0]
                mutado[d] += random.gauss(0, rango * 0.1 * tasa)
        return limitar_solucion(mutado, limites)
    
    for iteracion in range(max_iter):
        idx = np.argsort(afinidades)[::-1]
        pob_selecta = poblacion[idx][:n_select]
        af_selecta = afinidades[idx][:n_select]
        
        clones = []
        for i, (ab, af) in enumerate(zip(pob_selecta, af_selecta)):
            num_clones = max(1, int(beta * N / (i + 1)))
            tasa_mutacion = np.exp(-rho * af / (mejor_afinidad + 1))
            tasa_mutacion = max(0.01, min(0.5, tasa_mutacion))
            for _ in range(num_clones):
                clones.append(mutar(ab, tasa_mutacion))
        
        afinidad_clones = [evaluar_solucion(c, n_torres, ancho, alto, radio, puntos_demanda, demanda) for c in clones]
        
        todas_soluciones = list(pob_selecta) + clones
        todas_afinidades = list(af_selecta) + afinidad_clones
        idx_todas = np.argsort(todas_afinidades)[::-1]
        
        nueva_poblacion = [todas_soluciones[i] for i in idx_todas[:N-d]]
        nueva_afinidad = [todas_afinidades[i] for i in idx_todas[:N-d]]
        
        for _ in range(d):
            nuevo = generar_solucion_aleatoria(D, limites)
            nueva_poblacion.append(nuevo)
            nueva_afinidad.append(evaluar_solucion(nuevo, n_torres, ancho, alto, radio, puntos_demanda, demanda))
        
        poblacion = np.array(nueva_poblacion)
        afinidades = np.array(nueva_afinidad)
        
        mejor_idx = np.argmax(afinidades)
        if afinidades[mejor_idx] > mejor_afinidad:
            mejor_anticuerpo, mejor_afinidad = poblacion[mejor_idx].copy(), afinidades[mejor_idx]
        
        historial.append(mejor_afinidad)
        if (iteracion + 1) % 20 == 0:
            print(f"AIS - Iteración {iteracion+1}: {mejor_afinidad:.2f}")
    
    return mejor_anticuerpo, mejor_afinidad, historial

# ==================== VISUALIZACIÓN ====================
def visualizar_todas_soluciones(resultados, n_torres, ancho, alto, radio, puntos_demanda):
    """Muestra las 5 gráficas de torres en una sola ventana (2x3, un espacio vacío)"""
    nombres = ['PSO', 'GWO', 'AG', 'ABC', 'AIS']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, nombre in enumerate(nombres):
        ax = axes[idx]
        x = resultados[nombre][0]  # Posición de las torres
        fit = resultados[nombre][1]  # Aptitud
        torres = x.reshape(n_torres, 2)
        
        ax.set_xlim(0, ancho)
        ax.set_ylim(0, alto)
        ax.set_aspect('equal')
        
        # Puntos de demanda
        ax.scatter(puntos_demanda[:, 0], puntos_demanda[:, 1], c='blue', s=20, alpha=0.5)
        
        # Torres y cobertura
        for i, torre in enumerate(torres):
            circulo = plt.Circle(torre, radio, color='red', fill=False, linewidth=1.5, alpha=0.7)
            ax.add_patch(circulo)
            ax.scatter(torre[0], torre[1], c='red', s=100, marker='s', edgecolors='darkred', linewidths=2, zorder=5)
            ax.annotate(f'T{i+1}', torre, xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        ax.set_title(f'{nombre}\nAptitud = {fit:.2f}', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
    
    # Ocultar el subplot vacío (2,3 = 6 espacios, usamos 5)
    axes[5].axis('off')
    
    plt.suptitle('Optimización de Torres - Comparación de Algoritmos', fontsize=16)
    plt.tight_layout()
    plt.show()

def visualizar_convergencia(resultados):
    """Muestra la gráfica de convergencia"""
    plt.figure(figsize=(12, 6))
    colores = {'PSO': 'blue', 'GWO': 'green', 'AG': 'red', 'ABC': 'orange', 'AIS': 'purple'}
    
    for nombre, (_, _, hist) in resultados.items():
        plt.plot(hist, color=colores[nombre], label=nombre, linewidth=2)
    
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Aptitud (Cobertura)', fontsize=12)
    plt.title('Comparación de Convergencia de Algoritmos', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================== MAIN ====================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    # Generar puntos de demanda
    puntos_demanda = np.random.rand(N_PUNTOS_DEMANDA, 2)
    puntos_demanda[:, 0] *= ANCHO
    puntos_demanda[:, 1] *= ALTO
    demanda = np.ones(N_PUNTOS_DEMANDA)
    
    # Parámetros del problema
    D = N_TORRES * 2
    limites = [[0, ANCHO], [0, ALTO]] * N_TORRES
    limites = np.array(limites[:D])
    
    problema_params = (N_TORRES, ANCHO, ALTO, RADIO, puntos_demanda, demanda, D, limites)
    
    print("=" * 60)
    print("OPTIMIZACIÓN DE TORRES DE TELECOMUNICACIONES")
    print("=" * 60)
    print(f"Área: {ANCHO} x {ALTO}")
    print(f"Número de torres: {N_TORRES}")
    print(f"Radio de cobertura: {RADIO}")
    print(f"Población: {POBLACION}")
    print(f"Iteraciones: {MAX_ITER}")
    print("=" * 60)
    
    resultados = {}
    
    print("\n▶ Ejecutando PSO...")
    mejor_pos, mejor_val, hist = pso(problema_params)
    resultados['PSO'] = (mejor_pos, mejor_val, hist)
    print(f"  ✓ Mejor aptitud: {mejor_val:.2f}")
    
    print("\n▶ Ejecutando GWO...")
    mejor_pos, mejor_val, hist = gwo(problema_params)
    resultados['GWO'] = (mejor_pos, mejor_val, hist)
    print(f"  ✓ Mejor aptitud: {mejor_val:.2f}")
    
    print("\n▶ Ejecutando Algoritmo Genético...")
    mejor_pos, mejor_val, hist = ag(problema_params)
    resultados['AG'] = (mejor_pos, mejor_val, hist)
    print(f"  ✓ Mejor aptitud: {mejor_val:.2f}")
    
    print("\n▶ Ejecutando ABC...")
    mejor_pos, mejor_val, hist = abc(problema_params)
    resultados['ABC'] = (mejor_pos, mejor_val, hist)
    print(f"  ✓ Mejor aptitud: {mejor_val:.2f}")
    
    print("\n▶ Ejecutando AIS...")
    mejor_pos, mejor_val, hist = ais(problema_params)
    resultados['AIS'] = (mejor_pos, mejor_val, hist)
    print(f"  ✓ Mejor aptitud: {mejor_val:.2f}")
    
    print("\n" + "=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)
    for nombre, (_, fit, _) in resultados.items():
        print(f"{nombre}: {fit:.2f}")
    
    # Visualizaciones
    print("\n▶ Mostrando gráficos...")
    
    # Gráfica 1: Convergencia (1 ventana)
    visualizar_convergencia(resultados)
    
    # Gráfica 2: Todas las soluciones en UNA sola ventana (5 subplots)
    visualizar_todas_soluciones(resultados, N_TORRES, ANCHO, ALTO, RADIO, puntos_demanda)