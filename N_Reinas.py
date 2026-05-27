import random
import math
import time
from collections import deque
import heapq

class NReinas:
    def __init__(self, n):
        self.n = n
        self.estado_inicial = self.generar_estado_inicial()
        
    def generar_estado_inicial(self):
        estado = list(range(self.n))
        random.shuffle(estado)
        return estado
    
    def imprimir_tablero(self, estado, paso_num=None, ataques=None):
        """Imprime el tablero con las reinas"""
        if paso_num is not None:
            print(f"\n{'='*60}")
            print(f"PASO {paso_num}".center(60))
            print(f"{'='*60}")
            if ataques is None:
                ataques = self.contar_ataques(estado)
            print(f"Ataques: {ataques}")
        
        print("+" + "---+" * self.n)
        for i in range(self.n):
            fila = "|"
            for j in range(self.n):
                if estado[i] == j:
                    fila += " Q |"
                else:
                    fila += "   |"
            print(fila)
            print("+" + "---+" * self.n)
        print()
    
    def contar_ataques(self, estado):
        """Heurística: número de ataques entre reinas"""
        ataques = 0
        n = len(estado)
        for i in range(n):
            for j in range(i + 1, n):
                if estado[i] == estado[j] or abs(estado[i] - estado[j]) == j - i:
                    ataques += 1
        return ataques
    
    def es_solucion(self, estado):
        return self.contar_ataques(estado) == 0
    
    def generar_vecinos(self, estado):
        """Genera todos los vecinos moviendo una reina en su fila"""
        vecinos = []
        n = len(estado)
        for fila in range(n):
            for columna in range(n):
                if columna != estado[fila]:
                    vecino = estado.copy()
                    vecino[fila] = columna
                    vecinos.append(vecino)
        return vecinos
    
    def generar_vecino_aleatorio(self, estado):
        n = len(estado)
        fila = random.randint(0, n - 1)
        columna = random.randint(0, n - 1)
        while columna == estado[fila]:
            columna = random.randint(0, n - 1)
        vecino = estado.copy()
        vecino[fila] = columna
        return vecino
    
    # ==================== BFS ====================
    def bfs(self):
        print("\n" + "="*60)
        print("BFS (BÚSQUEDA EN ANCHURA)".center(60))
        print("="*60)
        
        nodo = self.estado_inicial.copy()
        
        if self.es_solucion(nodo):
            self.imprimir_tablero(nodo, 1)
            return nodo, [nodo], 0
        
        frontera = deque([nodo])
        explorados = []
        padres = {tuple(nodo): None}
        pasos = 0
        
        while frontera:
            nodo = frontera.popleft()
            pasos += 1
            
            ataques = self.contar_ataques(nodo)
            self.imprimir_tablero(nodo, pasos, ataques)
            
            if nodo in explorados:
                continue
            
            explorados.append(nodo)
            
            if self.es_solucion(nodo):
                camino = []
                actual = tuple(nodo)
                while actual:
                    camino.insert(0, list(actual))
                    actual = padres[actual]
                print(f"\n{'='*60}")
                print(f"✅ SOLUCIÓN ENCONTRADA EN {pasos} PASOS".center(60))
                print(f"{'='*60}")
                return nodo, camino, pasos
            
            for hijo in self.generar_vecinos(nodo):
                if hijo not in explorados and hijo not in frontera:
                    frontera.append(hijo)
                    padres[tuple(hijo)] = tuple(nodo)
        
        print("❌ No se encontró solución")
        return None, [], pasos
    
    # ==================== DFS ====================
    def dfs(self):
        print("\n" + "="*60)
        print("DFS (BÚSQUEDA EN PROFUNDIDAD)".center(60))
        print("="*60)
        
        nodo = self.estado_inicial.copy()
        
        if self.es_solucion(nodo):
            self.imprimir_tablero(nodo, 1)
            return nodo, [nodo], 0
        
        frontera = [nodo]
        explorados = []
        padres = {tuple(nodo): None}
        pasos = 0
        
        while frontera:
            nodo = frontera.pop()
            pasos += 1
            
            ataques = self.contar_ataques(nodo)
            self.imprimir_tablero(nodo, pasos, ataques)
            
            if nodo in explorados:
                continue
            
            explorados.append(nodo)
            
            if self.es_solucion(nodo):
                camino = []
                actual = tuple(nodo)
                while actual:
                    camino.insert(0, list(actual))
                    actual = padres[actual]
                print(f"\n{'='*60}")
                print(f"✅ SOLUCIÓN ENCONTRADA EN {pasos} PASOS".center(60))
                print(f"{'='*60}")
                return nodo, camino, pasos
            
            for hijo in self.generar_vecinos(nodo):
                if hijo not in explorados and hijo not in frontera:
                    frontera.append(hijo)
                    padres[tuple(hijo)] = tuple(nodo)
        
        print("❌ No se encontró solución")
        return None, [], pasos
    
    # ==================== LDFS ====================
    def ldfs(self, limite):
        print("\n" + "="*60)
        print(f"LDFS (LÍMITE: {limite})".center(60))
        print("="*60)
        
        nodo = self.estado_inicial.copy()
        
        if self.es_solucion(nodo):
            self.imprimir_tablero(nodo, 1)
            return nodo, [nodo], 0
        
        frontera = [nodo]
        explorados = []
        padres = {tuple(nodo): None}
        profundidades = {tuple(nodo): 0}
        pasos = 0
        
        while frontera:
            nodo = frontera.pop()
            pasos += 1
            prof_actual = profundidades.get(tuple(nodo), 0)
            
            ataques = self.contar_ataques(nodo)
            print(f"\n{'='*60}")
            print(f"PASO {pasos} (PROFUNDIDAD: {prof_actual})".center(60))
            print(f"{'='*60}")
            print(f"Ataques: {ataques}")
            self.imprimir_tablero(nodo, None, ataques)
            
            if nodo in explorados:
                continue
            
            explorados.append(nodo)
            
            if self.es_solucion(nodo):
                camino = []
                actual = tuple(nodo)
                while actual:
                    camino.insert(0, list(actual))
                    actual = padres[actual]
                print(f"\n{'='*60}")
                print(f"✅ SOLUCIÓN ENCONTRADA EN {pasos} PASOS".center(60))
                print(f"{'='*60}")
                return nodo, camino, pasos
            
            if prof_actual < limite:
                hijos = self.generar_vecinos(nodo)
                for hijo in reversed(hijos):
                    if hijo not in explorados and hijo not in frontera:
                        frontera.append(hijo)
                        padres[tuple(hijo)] = tuple(nodo)
                        profundidades[tuple(hijo)] = prof_actual + 1
        
        print("❌ No se encontró solución dentro del límite")
        return None, [], pasos
    
    # ==================== ILDFS ====================
    def ildfs(self):
        print("\n" + "="*60)
        print("ILDFS (BÚSQUEDA EN PROFUNDIDAD ITERATIVA)".center(60))
        print("="*60)
        
        limite = 1
        pasos_totales = 0
        
        while True:
            print(f"\n{'─'*60}")
            print(f"INTENTANDO CON LÍMITE = {limite}".center(60))
            print(f"{'─'*60}")
            
            resultado, camino, pasos = self.ldfs(limite)
            pasos_totales += pasos
            
            if resultado is not None:
                print(f"\n{'='*60}")
                print(f"✅ SOLUCIÓN ENCONTRADA".center(60))
                print(f"Límite final: {limite}".center(60))
                print(f"Pasos totales: {pasos_totales}".center(60))
                print(f"{'='*60}")
                return resultado, camino, pasos_totales
            
            limite += 1
    
    # ==================== VORAZ (GREEDY) ====================
    def voraz(self):
        print("\n" + "="*60)
        print("BÚSQUEDA VORAZ (GREEDY)".center(60))
        print("="*60)
        
        nodo = self.estado_inicial.copy()
        explorados = []
        camino = [nodo.copy()]
        pasos = 0
        
        while True:
            pasos += 1
            ataques = self.contar_ataques(nodo)
            self.imprimir_tablero(nodo, pasos, ataques)
            
            if self.es_solucion(nodo):
                print(f"\n{'='*60}")
                print(f"✅ SOLUCIÓN ENCONTRADA EN {pasos} PASOS".center(60))
                print(f"{'='*60}")
                return nodo, camino, pasos
            
            if nodo in explorados:
                print("⚠️ CICLO DETECTADO - No se encuentra solución")
                return None, camino, pasos
            
            explorados.append(nodo)
            
            vecinos = self.generar_vecinos(nodo)
            vecinos_con_ataques = [(self.contar_ataques(v), v) for v in vecinos]
            vecinos_con_ataques.sort(key=lambda x: x[0])
            
            mejor_nodo = None
            for _, vecino in vecinos_con_ataques:
                if vecino not in explorados:
                    mejor_nodo = vecino
                    break
            
            if mejor_nodo is None:
                print("❌ MÍNIMO LOCAL ALCANZADO - No se puede mejorar")
                return None, camino, pasos
            
            nodo = mejor_nodo
            camino.append(nodo.copy())
    
    # ==================== A* ====================
    def a_star(self):
        print("\n" + "="*60)
        print("A*".center(60))
        print("="*60)
        
        nodo = self.estado_inicial.copy()
        
        if self.es_solucion(nodo):
            self.imprimir_tablero(nodo, 1)
            return nodo, [nodo], 0
        
        frontera = []
        g_score = {tuple(nodo): 0}
        h_score = self.contar_ataques(nodo)
        f_score = {tuple(nodo): h_score}
        heapq.heappush(frontera, (f_score[tuple(nodo)], id(nodo), nodo))
        
        explorados = {}
        padres = {tuple(nodo): None}
        pasos = 0
        
        while frontera:
            _, _, nodo = heapq.heappop(frontera)
            pasos += 1
            
            ataques = self.contar_ataques(nodo)
            nodo_tuple = tuple(nodo)
            print(f"\n{'='*60}")
            print(f"PASO {pasos}".center(60))
            print(f"{'='*60}")
            print(f"Ataques: {ataques}")
            print(f"g(n) = {g_score.get(nodo_tuple, 0)}")
            print(f"h(n) = {ataques}")
            print(f"f(n) = {g_score.get(nodo_tuple, 0) + ataques}")
            self.imprimir_tablero(nodo, None, ataques)
            
            if nodo_tuple in explorados:
                continue
            
            explorados[nodo_tuple] = True
            
            if self.es_solucion(nodo):
                camino = []
                actual = nodo_tuple
                while actual:
                    camino.insert(0, list(actual))
                    actual = padres[actual]
                print(f"\n{'='*60}")
                print(f"✅ SOLUCIÓN ENCONTRADA EN {pasos} PASOS".center(60))
                print(f"{'='*60}")
                return nodo, camino, pasos
            
            for hijo in self.generar_vecinos(nodo):
                hijo_tuple = tuple(hijo)
                tentative_g = g_score.get(nodo_tuple, 0) + 1
                
                if hijo_tuple not in g_score or tentative_g < g_score[hijo_tuple]:
                    padres[hijo_tuple] = nodo_tuple
                    g_score[hijo_tuple] = tentative_g
                    f = tentative_g + self.contar_ataques(hijo)
                    f_score[hijo_tuple] = f
                    heapq.heappush(frontera, (f, id(hijo), hijo))
        
        print("❌ No se encontró solución")
        return None, [], pasos
    
    # ==================== BÚSQUEDA TABÚ ====================
    def busqueda_tabu(self, iteraciones_max=1000, tamano_tabu=20):
        print("\n" + "="*60)
        print("BÚSQUEDA TABÚ".center(60))
        print("="*60)
        
        nodo = self.estado_inicial.copy()
        mejor_solucion = nodo.copy()
        mejor_ataques = self.contar_ataques(nodo)
        lista_tabu = []
        camino = [nodo.copy()]
        iteracion = 0
        
        while iteracion < iteraciones_max:
            iteracion += 1
            ataques = self.contar_ataques(nodo)
            
            if iteracion % 10 == 0 or self.es_solucion(nodo):
                print(f"\n{'='*60}")
                print(f"ITERACIÓN {iteracion}".center(60))
                print(f"{'='*60}")
                print(f"Ataques: {ataques}")
                if len(lista_tabu) > 0:
                    print(f"Tamaño lista tabú: {len(lista_tabu)}")
                self.imprimir_tablero(nodo, None, ataques)
            
            if self.es_solucion(nodo):
                print(f"\n{'='*60}")
                print(f"✅ SOLUCIÓN ENCONTRADA EN {iteracion} ITERACIONES".center(60))
                print(f"{'='*60}")
                return nodo, camino, iteracion
            
            vecinos = self.generar_vecinos(nodo)
            vecinos_con_ataques = [(self.contar_ataques(v), v) for v in vecinos]
            vecinos_con_ataques.sort(key=lambda x: x[0])
            
            mejor_vecino = None
            for _, vecino in vecinos_con_ataques:
                if vecino not in lista_tabu:
                    mejor_vecino = vecino
                    break
            
            if mejor_vecino is None:
                print("⚠️ No hay vecinos válidos (todos son tabú)")
                break
            
            nodo = mejor_vecino
            camino.append(nodo.copy())
            
            ataques_actual = self.contar_ataques(nodo)
            if ataques_actual < mejor_ataques:
                mejor_solucion = nodo.copy()
                mejor_ataques = ataques_actual
                print(f"✨ NUEVA MEJOR SOLUCIÓN: {mejor_ataques} ataques")
            
            lista_tabu.append(nodo.copy())
            if len(lista_tabu) > tamano_tabu:
                lista_tabu.pop(0)
        
        print(f"\n📊 Mejor solución encontrada: {mejor_ataques} ataques")
        self.imprimir_tablero(mejor_solucion, None, mejor_ataques)
        return mejor_solucion, camino, iteracion
    
    # ==================== RECOCIDO SIMULADO ====================
    def recocido_simulado(self, temp_inicial=1000, enfriamiento=0.95, selecciones_max=100):
        print("\n" + "="*60)
        print("RECOCIDO SIMULADO".center(60))
        print("="*60)
        
        nodo = self.estado_inicial.copy()
        mejor_solucion = nodo.copy()
        mejor_ataques = self.contar_ataques(nodo)
        temperatura = temp_inicial
        camino = [nodo.copy()]
        iteracion = 0
        
        print(f"Temperatura inicial: {temperatura}")
        print(f"Factor de enfriamiento: {enfriamiento}")
        print(f"Iteraciones por temperatura: {selecciones_max}")
        
        while temperatura > 0.01:
            for _ in range(selecciones_max):
                iteracion += 1
                ataques = self.contar_ataques(nodo)
                
                if iteracion % 50 == 0 or self.es_solucion(nodo):
                    print(f"\n{'='*60}")
                    print(f"ITERACIÓN {iteracion}".center(60))
                    print(f"{'='*60}")
                    print(f"Temperatura: {temperatura:.2f}")
                    print(f"Ataques: {ataques}")
                    self.imprimir_tablero(nodo, None, ataques)
                
                if self.es_solucion(nodo):
                    print(f"\n{'='*60}")
                    print(f"✅ SOLUCIÓN ENCONTRADA EN {iteracion} ITERACIONES".center(60))
                    print(f"{'='*60}")
                    return nodo, camino, iteracion
                
                hijo = self.generar_vecino_aleatorio(nodo)
                delta_e = self.contar_ataques(hijo) - self.contar_ataques(nodo)
                
                if delta_e < 0:
                    nodo = hijo
                    camino.append(nodo.copy())
                    if self.contar_ataques(nodo) < mejor_ataques:
                        mejor_solucion = nodo.copy()
                        mejor_ataques = self.contar_ataques(nodo)
                        print(f"  ✨ NUEVO MÍNIMO: {mejor_ataques} ataques")
                else:
                    probabilidad = math.exp(-delta_e / temperatura)
                    if random.random() < probabilidad:
                        nodo = hijo
                        camino.append(nodo.copy())
                        print(f"  🌡️ Aceptado peor solución (Δ={delta_e}, P={probabilidad:.3f})")
            
            temperatura *= enfriamiento
            print(f"\n🌡️ TEMPERATURA ENFRIADA: {temperatura:.2f}")
        
        print(f"\n📊 Mejor solución encontrada: {mejor_ataques} ataques")
        self.imprimir_tablero(mejor_solucion, None, mejor_ataques)
        return mejor_solucion, camino, iteracion


def menu():
    while True:
        print("\n" + "="*60)
        print("PROBLEMA DE LAS N REINAS".center(60))
        print("="*60)
        print("\n🎯 Seleccione el algoritmo a ejecutar:")
        print("-"*40)
        print("1. BFS (Búsqueda en Anchura)")
        print("2. DFS (Búsqueda en Profundidad)")
        print("3. LDFS (Búsqueda en Profundidad Limitada)")
        print("4. ILDFS (Búsqueda en Profundidad Iterativa)")
        print("5. Búsqueda Voraz (Greedy)")
        print("6. A*")
        print("7. Búsqueda Tabú")
        print("8. Recocido Simulado")
        print("0. Salir")
        print("="*60)
        
        try:
            opcion = int(input("\n👉 Opción: "))
            if opcion == 0:
                print("\n👋 ¡Hasta luego!")
                break
            
            if opcion < 1 or opcion > 8:
                print("\n❌ Opción no válida")
                input("Presione Enter para continuar...")
                continue
            
            n = int(input("\n📏 Ingrese el tamaño del tablero (N): "))
            if n < 4:
                print("\n⚠️ Para N < 4 no hay solución. Usando N=4")
                n = 4
            
            problema = NReinas(n)
            
            print(f"\n{'='*60}")
            print(f"ESTADO INICIAL (N={n})".center(60))
            print(f"{'='*60}")
            print(f"Ataques: {problema.contar_ataques(problema.estado_inicial)}")
            problema.imprimir_tablero(problema.estado_inicial, None)
            
            input("\n⏯️ Presione Enter para comenzar la búsqueda...")
            
            inicio = time.time()
            
            if opcion == 1:
                solucion, camino, pasos = problema.bfs()
            elif opcion == 2:
                solucion, camino, pasos = problema.dfs()
            elif opcion == 3:
                limite = int(input("Ingrese el límite de profundidad: "))
                solucion, camino, pasos = problema.ldfs(limite)
            elif opcion == 4:
                solucion, camino, pasos = problema.ildfs()
            elif opcion == 5:
                solucion, camino, pasos = problema.voraz()
            elif opcion == 6:
                solucion, camino, pasos = problema.a_star()
            elif opcion == 7:
                max_iter = int(input("Iteraciones máximas (default 1000): ") or 1000)
                tabu_size = int(input("Tamaño lista tabú (default 20): ") or 20)
                solucion, camino, pasos = problema.busqueda_tabu(max_iter, tabu_size)
            elif opcion == 8:
                temp = float(input("Temperatura inicial (default 1000): ") or 1000)
                enfriamiento = float(input("Factor de enfriamiento (default 0.95): ") or 0.95)
                selecciones = int(input("Iteraciones por temperatura (default 100): ") or 100)
                solucion, camino, pasos = problema.recocido_simulado(temp, enfriamiento, selecciones)
            
            fin = time.time()
            
            print("\n" + "="*60)
            print("RESUMEN FINAL".center(60))
            print("="*60)
            print(f"⏱️ Tiempo de ejecución: {fin - inicio:.4f} segundos")
            print(f"📝 Pasos totales: {pasos}")
            
            if solucion and problema.es_solucion(solucion):
                print(f"✅ SOLUCIÓN COMPLETA ENCONTRADA")
                print(f"Ataques finales: 0")
                print(f"Configuración final: {solucion}")
            elif solucion:
                print(f"⚠️ Mejor solución encontrada con {problema.contar_ataques(solucion)} ataques")
                print(f"Configuración: {solucion}")
            else:
                print("❌ No se encontró solución")
            
            input("\nPresione Enter para continuar...")
            
        except ValueError as e:
            print(f"\n❌ Error: {e}")
            input("Presione Enter para continuar...")
        except KeyboardInterrupt:
            print("\n\n👋 Programa terminado por el usuario")
            break


if __name__ == "__main__":
    random.seed(42)  # Para reproducibilidad
    menu()