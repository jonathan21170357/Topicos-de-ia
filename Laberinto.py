import random
import math
import time
import heapq
from collections import deque

class Laberinto:
    def __init__(self, m, n, porcentaje_obstaculos):
        self.m = m
        self.n = n
        self.inicio = (0, 0)
        self.meta = (m-1, n-1)
        self.obstaculos = set()
        self.generar_obstaculos(porcentaje_obstaculos)
        
    def generar_obstaculos(self, porcentaje):
        """Genera obstáculos aleatoriamente según el porcentaje especificado"""
        total_celdas = self.m * self.n
        num_obstaculos = int(total_celdas * (porcentaje / 100))
        
        posibles_obstaculos = []
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) != self.inicio and (i, j) != self.meta:
                    posibles_obstaculos.append((i, j))
        
        if num_obstaculos > len(posibles_obstaculos):
            num_obstaculos = len(posibles_obstaculos)
            
        self.obstaculos = set(random.sample(posibles_obstaculos, num_obstaculos))
    
    def es_valido(self, posicion):
        """Verifica si una posición es válida en el laberinto"""
        i, j = posicion
        return (0 <= i < self.m and 0 <= j < self.n and 
                posicion not in self.obstaculos)
    
    def obtener_vecinos(self, posicion):
        """Obtiene los vecinos válidos de una posición"""
        i, j = posicion
        movimientos = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        return [mov for mov in movimientos if self.es_valido(mov)]
    
    def distancia_manhattan(self, pos1, pos2):
        """Calcula la distancia Manhattan entre dos posiciones"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def mostrar(self, camino=None, visitados=None, paso_actual=None):
        """Muestra el laberinto en consola"""
        print("\n" + "=" * (self.n * 4 + 2))
        for i in range(self.m):
            fila = ""
            for j in range(self.n):
                if (i, j) == self.inicio:
                    fila += " I "
                elif (i, j) == self.meta:
                    fila += " M "
                elif (i, j) in self.obstaculos:
                    fila += " ■ "
                elif camino and (i, j) in camino:
                    if paso_actual and (i, j) == paso_actual:
                        fila += " ● "
                    else:
                        fila += " ○ "
                elif visitados and (i, j) in visitados:
                    fila += " · "
                else:
                    fila += "   "
                fila += "|"
            print(fila)
            print("-" * (self.n * 4 + 2))


class Buscador:
    def __init__(self, laberinto):
        self.laberinto = laberinto
        self.visitados = set()
        self.camino = []
        self.pasos_simulacion = []
        self.profundidad_actual = 0
    
    def registrar_paso(self, posicion_actual, frontera, visitados, profundidad=0):
        """Registra un paso de la simulación"""
        self.pasos_simulacion.append({
            'actual': posicion_actual,
            'frontera': frontera.copy() if frontera else [],
            'visitados': visitados.copy(),
            'profundidad': profundidad
        })
    
    def mostrar_paso_a_paso(self, delay=0.3):
        """Muestra la simulación paso a paso"""
        for i, paso in enumerate(self.pasos_simulacion):
            print(f"\n{'='*60}")
            print(f"PASO {i+1}: Explorando posición {paso['actual']}")
            print(f"Profundidad: {paso['profundidad']}")
            if paso['frontera']:
                print(f"Frontera: {paso['frontera'][:10]}..." if len(paso['frontera']) > 10 else f"Frontera: {paso['frontera']}")
            print(f"Celdas visitadas: {len(paso['visitados'])}")
            self.laberinto.mostrar(
                camino=[paso['actual']], 
                visitados=paso['visitados'],
                paso_actual=paso['actual']
            )
            time.sleep(delay)
    
    # ==================== BFS ====================
    def bfs(self, mostrar_pasos=True):
        """Búsqueda en Anchura (BFS) - Según pseudocódigo"""
        print("\n" + "="*60)
        print("=== BFS (Búsqueda en Anchura) ===")
        print("Estrategia: Cola FIFO - Explora por niveles")
        print("="*60)
        
        self.pasos_simulacion = []
        self.visitados = set()
        
        frontera = deque()
        frontera.append((self.laberinto.inicio, [self.laberinto.inicio], 0))
        
        while True:
            if not frontera:
                return None
            
            posicion_actual, camino_actual, profundidad = frontera.popleft()
            
            if posicion_actual in self.visitados:
                continue
            
            self.visitados.add(posicion_actual)
            
            if mostrar_pasos:
                self.registrar_paso(posicion_actual, [p[0] for p in frontera], self.visitados, profundidad)
            
            if posicion_actual == self.laberinto.meta:
                self.camino = camino_actual
                return camino_actual
            
            for vecino in self.laberinto.obtener_vecinos(posicion_actual):
                if vecino not in self.visitados:
                    frontera.append((vecino, camino_actual + [vecino], profundidad + 1))
    
    # ==================== DFS ====================
    def dfs(self, mostrar_pasos=True):
        """Búsqueda en Profundidad (DFS) - Según pseudocódigo"""
        print("\n" + "="*60)
        print("=== DFS (Búsqueda en Profundidad) ===")
        print("Estrategia: Pila FILO - Explora hasta el fondo")
        print("="*60)
        
        self.pasos_simulacion = []
        self.visitados = set()
        
        frontera = [(self.laberinto.inicio, [self.laberinto.inicio], 0)]
        
        while True:
            if not frontera:
                return None
            
            posicion_actual, camino_actual, profundidad = frontera.pop()
            
            if posicion_actual in self.visitados:
                continue
            
            self.visitados.add(posicion_actual)
            
            if mostrar_pasos:
                self.registrar_paso(posicion_actual, [p[0] for p in frontera], self.visitados, profundidad)
            
            if posicion_actual == self.laberinto.meta:
                self.camino = camino_actual
                return camino_actual
            
            for vecino in self.laberinto.obtener_vecinos(posicion_actual):
                if vecino not in self.visitados:
                    frontera.append((vecino, camino_actual + [vecino], profundidad + 1))
    
    # ==================== LDFS ====================
    def ldfs(self, limite, mostrar_pasos=True):
        """Búsqueda en Profundidad Limitada (LDFS) - Según pseudocódigo"""
        print(f"\n=== LDFS (Búsqueda en Profundidad Limitada) - Límite: {limite} ===")
        
        self.pasos_simulacion = []
        self.visitados = set()
        
        frontera = [(self.laberinto.inicio, [self.laberinto.inicio], 0)]
        
        while True:
            if not frontera:
                return None
            
            posicion_actual, camino_actual, profundidad = frontera.pop()
            
            if posicion_actual in self.visitados:
                continue
            
            self.visitados.add(posicion_actual)
            
            if mostrar_pasos:
                self.registrar_paso(posicion_actual, [p[0] for p in frontera], self.visitados, profundidad)
            
            if posicion_actual == self.laberinto.meta:
                self.camino = camino_actual
                return camino_actual
            
            if profundidad < limite:
                for vecino in self.laberinto.obtener_vecinos(posicion_actual):
                    if vecino not in self.visitados:
                        frontera.append((vecino, camino_actual + [vecino], profundidad + 1))
        
        return None
    
    # ==================== ILDFS ====================
    def ildfs(self, max_limite=50, mostrar_pasos=True):
        """Búsqueda en Profundidad Iterativa (ILDFS) - Según pseudocódigo"""
        print("\n" + "="*60)
        print("=== ILDFS (Búsqueda en Profundidad Iterativa) ===")
        print("Estrategia: Aumenta el límite progresivamente")
        print("="*60)
        
        limite = 1
        
        while limite <= max_limite:
            print(f"\n--- Intentando con límite = {limite} ---")
            resultado = self.ldfs(limite, mostrar_pasos=False)
            
            if resultado:
                if mostrar_pasos:
                    self.mostrar_paso_a_paso(0.2)
                return resultado
            else:
                limite += 1
        
        return None
    
    # ==================== VORAZ (Greedy) ====================
    def voraz(self, mostrar_pasos=True):
        """Búsqueda Voraz (Greedy) - Según pseudocódigo"""
        print("\n" + "="*60)
        print("=== Búsqueda Voraz (Greedy) ===")
        print("Estrategia: Solo usa la heurística (distancia a la meta)")
        print("Heurística: Distancia Manhattan")
        print("="*60)
        
        self.pasos_simulacion = []
        self.visitados = set()
        
        nodo_actual = self.laberinto.inicio
        camino_actual = [nodo_actual]
        
        while True:
            if nodo_actual is None:
                return None
            
            if nodo_actual in self.visitados:
                nodo_actual = None
                continue
            
            self.visitados.add(nodo_actual)
            
            if mostrar_pasos:
                self.registrar_paso(nodo_actual, [], self.visitados, len(camino_actual)-1)
            
            if nodo_actual == self.laberinto.meta:
                self.camino = camino_actual
                return camino_actual
            
            vecinos = self.laberinto.obtener_vecinos(nodo_actual)
            
            if not vecinos:
                return None
            
            hijos_evaluados = []
            for vecino in vecinos:
                if vecino not in self.visitados:
                    heuristica = self.laberinto.distancia_manhattan(vecino, self.laberinto.meta)
                    hijos_evaluados.append((heuristica, vecino))
            
            if not hijos_evaluados:
                return None
            
            hijos_evaluados.sort(key=lambda x: x[0])
            mejor_vecino = hijos_evaluados[0][1]
            camino_actual.append(mejor_vecino)
            nodo_actual = mejor_vecino
    
    # ==================== A* ====================
    def a_star(self, mostrar_pasos=True):
        """Algoritmo A* - Según pseudocódigo"""
        print("\n" + "="*60)
        print("=== A* (A Star) ===")
        print("Estrategia: f(n) = g(n) + h(n)")
        print("g(n): costo real | h(n): distancia Manhattan")
        print("="*60)
        
        self.pasos_simulacion = []
        self.visitados = set()
        
        frontera = []
        heapq.heappush(frontera, (0, self.laberinto.inicio, [self.laberinto.inicio], 0))
        costos_reales = {self.laberinto.inicio: 0}
        
        while True:
            if not frontera:
                return None
            
            f_actual, posicion_actual, camino_actual, g_actual = heapq.heappop(frontera)
            
            if posicion_actual in self.visitados:
                continue
            
            self.visitados.add(posicion_actual)
            
            if mostrar_pasos:
                self.registrar_paso(posicion_actual, [p[1] for p in frontera], self.visitados, len(camino_actual)-1)
            
            if posicion_actual == self.laberinto.meta:
                self.camino = camino_actual
                return camino_actual
            
            for vecino in self.laberinto.obtener_vecinos(posicion_actual):
                if vecino not in self.visitados:
                    g_nuevo = g_actual + 1
                    
                    if vecino not in costos_reales or g_nuevo < costos_reales[vecino]:
                        costos_reales[vecino] = g_nuevo
                        h = self.laberinto.distancia_manhattan(vecino, self.laberinto.meta)
                        f_nuevo = g_nuevo + h
                        heapq.heappush(frontera, (f_nuevo, vecino, camino_actual + [vecino], g_nuevo))
    
    # ==================== BÚSQUEDA TABÚ ====================
    def busqueda_tabu(self, max_iteraciones=50, tamano_tabu=10, mostrar_pasos=True):
        """Búsqueda Tabú - Según pseudocódigo"""
        print("\n" + "="*60)
        print("=== Búsqueda Tabú ===")
        print(f"Estrategia: Lista Tabú (tamaño={tamano_tabu})")
        print("Evita ciclos usando memoria de corto plazo")
        print("="*60)
        
        self.pasos_simulacion = []
        
        nodo_actual = self.laberinto.inicio
        mejor_solucion = nodo_actual
        mejor_camino = [nodo_actual]
        mejor_distancia = self.laberinto.distancia_manhattan(nodo_actual, self.laberinto.meta)
        
        lista_tabu = []
        iteracion = 0
        camino_actual = [nodo_actual]
        
        while iteracion < max_iteraciones and nodo_actual != self.laberinto.meta:
            iteracion += 1
            
            if mostrar_pasos and iteracion % 5 == 0:
                self.registrar_paso(nodo_actual, lista_tabu.copy(), set([nodo_actual]), iteracion)
            
            vecinos = self.laberinto.obtener_vecinos(nodo_actual)
            
            hijos_evaluados = []
            for vecino in vecinos:
                if vecino not in lista_tabu:
                    heuristica = self.laberinto.distancia_manhattan(vecino, self.laberinto.meta)
                    hijos_evaluados.append((heuristica, vecino))
            
            if not hijos_evaluados:
                for vecino in vecinos:
                    heuristica = self.laberinto.distancia_manhattan(vecino, self.laberinto.meta)
                    hijos_evaluados.append((heuristica, vecino))
            
            if not hijos_evaluados:
                break
            
            hijos_evaluados.sort(key=lambda x: x[0])
            mejor_vecino = hijos_evaluados[0][1]
            nodo_actual = mejor_vecino
            camino_actual.append(nodo_actual)
            
            distancia_actual = self.laberinto.distancia_manhattan(nodo_actual, self.laberinto.meta)
            if distancia_actual < mejor_distancia:
                mejor_solucion = nodo_actual
                mejor_distancia = distancia_actual
                mejor_camino = camino_actual.copy()
            
            lista_tabu.append(nodo_actual)
            
            if len(lista_tabu) > tamano_tabu:
                lista_tabu.pop(0)
        
        if nodo_actual == self.laberinto.meta:
            self.camino = camino_actual
        else:
            self.camino = mejor_camino
        
        return self.camino
    
    # ==================== RECOCIDO SIMULADO ====================
    def recocido_simulado(self, temp_inicial=100, enfriamiento=0.95, 
                          selecciones_max=20, mostrar_pasos=True):
        """Recocido Simulado - Según pseudocódigo"""
        print("\n" + "="*60)
        print("=== Recocido Simulado ===")
        print(f"Estrategia: Temperatura inicial={temp_inicial}, Enfriamiento={enfriamiento}")
        print("Acepta soluciones peores con cierta probabilidad")
        print("="*60)
        
        self.pasos_simulacion = []
        
        # nodo = estado inicial
        nodo_actual = self.laberinto.inicio
        # mejor_solucion = nodo
        mejor_solucion = nodo_actual
        mejor_camino = [nodo_actual]
        # mejor_distancia INICIALIZADA correctamente
        mejor_distancia = self.laberinto.distancia_manhattan(nodo_actual, self.laberinto.meta)
        # temperatura = temp_inicial
        temperatura = temp_inicial
        camino_actual = [nodo_actual]
        iteracion = 0
        
        # while temperatura > 0.01
        while temperatura > 0.01 and nodo_actual != self.laberinto.meta:
            # for i = 1 to selecciones_max
            for _ in range(selecciones_max):
                if mostrar_pasos and iteracion % 5 == 0:
                    self.registrar_paso(nodo_actual, [], set([nodo_actual]), iteracion)
                
                # generate_random_solution(nodo)
                vecinos = self.laberinto.obtener_vecinos(nodo_actual)
                if not vecinos:
                    break
                
                hijo = random.choice(vecinos)
                
                # delta_e = Evaluar(hijo) - Evaluar(nodo)
                dist_actual = self.laberinto.distancia_manhattan(nodo_actual, self.laberinto.meta)
                dist_hijo = self.laberinto.distancia_manhattan(hijo, self.laberinto.meta)
                delta_e = dist_hijo - dist_actual
                
                # if delta_e < 0 (hijo es mejor)
                if delta_e < 0:
                    nodo_actual = hijo
                    camino_actual.append(nodo_actual)
                    # if Evaluar(nodo) < Evaluar(mejor_solucion)
                    if dist_hijo < mejor_distancia:
                        mejor_solucion = nodo_actual
                        mejor_distancia = dist_hijo
                        mejor_camino = camino_actual.copy()
                else:
                    # probabilidad = exp(-delta_e / temperatura)
                    probabilidad = math.exp(-delta_e / temperatura)
                    # if Random(0,1) < probabilidad
                    if random.random() < probabilidad:
                        nodo_actual = hijo
                        camino_actual.append(nodo_actual)
                
                if nodo_actual == self.laberinto.meta:
                    break
            
            iteracion += 1
            # temperatura = temperatura * enfriamiento
            temperatura *= enfriamiento
        
        if nodo_actual == self.laberinto.meta:
            self.camino = camino_actual
        else:
            self.camino = mejor_camino
        
        return self.camino


class Menu:
    def __init__(self):
        self.laberinto = None
        self.buscador = None
    
    def crear_laberinto(self):
        """Crea un nuevo laberinto"""
        print("\n" + "="*60)
        print("=== CREAR LABERINTO ===")
        print("="*60)
        
        while True:
            try:
                m = int(input("Ingrese el número de filas (m): "))
                n = int(input("Ingrese el número de columnas (n): "))
                porcentaje = float(input("Ingrese el porcentaje de obstáculos (0-100): "))
                
                if m <= 0 or n <= 0:
                    print("Error: Las dimensiones deben ser positivas")
                    continue
                    
                if porcentaje < 0 or porcentaje > 100:
                    print("Error: El porcentaje debe estar entre 0 y 100")
                    continue
                
                self.laberinto = Laberinto(m, n, porcentaje)
                self.buscador = Buscador(self.laberinto)
                
                print("\n" + "="*60)
                print("LABERINTO CREADO EXITOSAMENTE:")
                print(f"Dimensiones: {m} x {n}")
                print(f"Obstáculos: {porcentaje}% ({len(self.laberinto.obstaculos)} celdas)")
                print("="*60)
                self.laberinto.mostrar()
                return
                
            except ValueError:
                print("Error: Ingrese valores numéricos válidos")
    
    def ejecutar_algoritmo(self, algoritmo):
        """Ejecuta el algoritmo seleccionado"""
        if not self.laberinto:
            print("Primero debe crear un laberinto (Opción 1)")
            return
        
        print("\n" + "="*60)
        print(f"EJECUTANDO ALGORITMO: {algoritmo}")
        print("="*60)
        
        inicio = time.time()
        
        if algoritmo == "BFS":
            camino = self.buscador.bfs(mostrar_pasos=True)
        elif algoritmo == "DFS":
            camino = self.buscador.dfs(mostrar_pasos=True)
        elif algoritmo == "LDFS":
            limite = int(input("Ingrese el límite de profundidad: "))
            camino = self.buscador.ldfs(limite, mostrar_pasos=True)
        elif algoritmo == "ILDFS":
            max_limite = int(input("Ingrese el límite máximo (recomendado 30-50): "))
            camino = self.buscador.ildfs(max_limite, mostrar_pasos=True)
        elif algoritmo == "Voraz":
            camino = self.buscador.voraz(mostrar_pasos=True)
        elif algoritmo == "A*":
            camino = self.buscador.a_star(mostrar_pasos=True)
        elif algoritmo == "Búsqueda Tabú":
            iteraciones = int(input("Iteraciones máximas (recomendado 50-100): "))
            camino = self.buscador.busqueda_tabu(max_iteraciones=iteraciones, mostrar_pasos=True)
        elif algoritmo == "Recocido Simulado":
            temp = float(input("Temperatura inicial (recomendado 100): "))
            enf = float(input("Factor de enfriamiento (recomendado 0.95): "))
            camino = self.buscador.recocido_simulado(temp_inicial=temp, enfriamiento=enf, mostrar_pasos=True)
        
        fin = time.time()
        tiempo_ejecucion = fin - inicio
        
        print("\n" + "="*60)
        print("RESULTADOS:")
        print("="*60)
        
        if camino:
            print(f"✓ ¡SOLUCIÓN ENCONTRADA!")
            print(f"  Tiempo de ejecución: {tiempo_ejecucion:.4f} segundos")
            print(f"  Longitud del camino: {len(camino)} pasos")
            print(f"  Camino: {camino}")
            print("\nLaberinto con la solución encontrada:")
            self.laberinto.mostrar(camino=camino)
            
            if len(self.buscador.pasos_simulacion) > 0:
                print("\n" + "="*60)
                ver_pasos = input("¿Ver simulación paso a paso? (s/n): ").lower()
                if ver_pasos == 's':
                    delay = float(input("Velocidad (segundos entre pasos, ej: 0.3): "))
                    self.buscador.mostrar_paso_a_paso(delay)
        else:
            print(f"✗ NO se encontró solución")
            print(f"  Tiempo de ejecución: {tiempo_ejecucion:.4f} segundos")
            print("  Posible causa: Obstáculos bloquean el camino hacia la meta")
    
    def mostrar_menu(self):
        """Muestra el menú principal"""
        while True:
            print("\n" + "="*60)
            print("     SISTEMA DE RESOLUCIÓN DE LABERINTOS")
            print("="*60)
            print("  ALGORITMOS DE BÚSQUEDA DISPONIBLES:")
            print("-"*60)
            print("  1. Crear / Configurar Laberinto")
            print("  2. BFS - Búsqueda en Anchura")
            print("  3. DFS - Búsqueda en Profundidad")
            print("  4. LDFS - Búsqueda en Profundidad Limitada")
            print("  5. ILDFS - Búsqueda en Profundidad Iterativa")
            print("  6. Voraz (Greedy) - Heurística Manhattan")
            print("  7. A* (A Star) - Óptimo y Rápido")
            print("  8. Búsqueda Tabú - Con memoria")
            print("  9. Recocido Simulado - Metaheurística")
            print("  0. Salir")
            print("="*60)
            
            opcion = input("Seleccione una opción (0-9): ")
            
            if opcion == "1":
                self.crear_laberinto()
            elif opcion == "2":
                self.ejecutar_algoritmo("BFS")
            elif opcion == "3":
                self.ejecutar_algoritmo("DFS")
            elif opcion == "4":
                self.ejecutar_algoritmo("LDFS")
            elif opcion == "5":
                self.ejecutar_algoritmo("ILDFS")
            elif opcion == "6":
                self.ejecutar_algoritmo("Voraz")
            elif opcion == "7":
                self.ejecutar_algoritmo("A*")
            elif opcion == "8":
                self.ejecutar_algoritmo("Búsqueda Tabú")
            elif opcion == "9":
                self.ejecutar_algoritmo("Recocido Simulado")
            elif opcion == "0":
                print("\n" + "="*60)
                print("¡Gracias por usar el sistema!")
                print("="*60)
                break
            else:
                print("Opción no válida. Intente de nuevo.")
            
            input("\nPresione Enter para continuar...")


def main():
    """Función principal"""
    print("\n" + "="*60)
    print("   BIENVENIDO AL SISTEMA DE RESOLUCIÓN DE LABERINTOS")
    print("="*60)
    print("\nEste programa implementa 7 algoritmos de búsqueda:")
    print("  • BFS, DFS, LDFS, ILDFS (Búsquedas ciegas)")
    print("  • Voraz y A* (Búsquedas heurísticas)")
    print("  • Búsqueda Tabú y Recocido Simulado (Metaheurísticas)")
    print("\nCaracterísticas:")
    print("  • Generación aleatoria de obstáculos")
    print("  • Visualización paso a paso")
    print("  • Medición de tiempos de ejecución")
    
    menu = Menu()
    menu.mostrar_menu()


if __name__ == "__main__":
    main()