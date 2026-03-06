import random
import time
import heapq
from collections import deque

# ==============================
# GENERAR TABLERO
# ==============================

def generar_tablero(m, n, porcentaje_obstaculos):
    tablero = [[" " for _ in range(n)] for _ in range(m)]

    total_celdas = m * n
    obstaculos = int(total_celdas * porcentaje_obstaculos / 100)

    colocados = 0
    while colocados < obstaculos:
        x = random.randint(0, m-1)
        y = random.randint(0, n-1)
        if tablero[x][y] == " ":
            tablero[x][y] = "#"
            colocados += 1

    # Inicio y meta
    tablero[0][0] = "S"
    tablero[m-1][n-1] = "G"

    return tablero


# ==============================
# MOSTRAR TABLERO
# ==============================

def mostrar_tablero(tablero):
    for fila in tablero:
        print("".join(fila))
    print()


# ==============================
# MOVIMIENTOS
# ==============================

def obtener_movimientos():
    print("Ingrese movimientos permitidos:")
    print("1 - 4 direcciones (↑↓←→)")
    print("2 - 8 direcciones (incluye diagonales)")
    opcion = int(input("Opción: "))

    if opcion == 1:
        return [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        return [(-1,0),(1,0),(0,-1),(0,1),
                (-1,-1),(-1,1),(1,-1),(1,1)]


# ==============================
# FUNCIONES AUXILIARES
# ==============================

def es_valido(tablero, x, y):
    return (0 <= x < len(tablero) and
            0 <= y < len(tablero[0]) and
            tablero[x][y] != "#")


def reconstruir_camino(padres, meta):
    camino = []
    nodo = meta
    while nodo:
        camino.append(nodo)
        nodo = padres[nodo]
    camino.reverse()
    return camino


# ==============================
# BFS
# ==============================

def bfs(tablero, movimientos):
    inicio = (0,0)
    meta = (len(tablero)-1, len(tablero[0])-1)

    frontera = deque([inicio])
    padres = {inicio: None}
    visitados = set()

    while frontera:
        actual = frontera.popleft()
        visitados.add(actual)

        print("Explorando:", actual)
        time.sleep(0.1)

        if actual == meta:
            return reconstruir_camino(padres, meta)

        for dx, dy in movimientos:
            nx, ny = actual[0] + dx, actual[1] + dy
            if es_valido(tablero, nx, ny):
                vecino = (nx, ny)
                if vecino not in visitados and vecino not in frontera:
                    frontera.append(vecino)
                    padres[vecino] = actual

    return None


# ==============================
# DFS
# ==============================

def dfs(tablero, movimientos):
    inicio = (0,0)
    meta = (len(tablero)-1, len(tablero[0])-1)

    frontera = [inicio]
    padres = {inicio: None}
    visitados = set()

    while frontera:
        actual = frontera.pop()
        visitados.add(actual)

        print("Explorando:", actual)
        time.sleep(0.1)

        if actual == meta:
            return reconstruir_camino(padres, meta)

        for dx, dy in movimientos:
            nx, ny = actual[0] + dx, actual[1] + dy
            if es_valido(tablero, nx, ny):
                vecino = (nx, ny)
                if vecino not in visitados:
                    frontera.append(vecino)
                    padres[vecino] = actual

    return None

def ldfs(tablero, movimientos, limite):
    inicio = (0,0)
    meta = (len(tablero)-1, len(tablero[0])-1)

    def dfs_limitado(nodo, profundidad, visitados, padres):
        print("Explorando:", nodo, "Prof:", profundidad)
        time.sleep(0.05)

        if nodo == meta:
            return reconstruir_camino(padres, nodo)

        if profundidad >= limite:
            return None

        visitados.add(nodo)

        for dx, dy in movimientos:
            nx, ny = nodo[0] + dx, nodo[1] + dy
            vecino = (nx, ny)
            if es_valido(tablero, nx, ny) and vecino not in visitados:
                padres[vecino] = nodo
                resultado = dfs_limitado(vecino, profundidad+1, visitados, padres)
                if resultado:
                    return resultado

        return None

    return dfs_limitado(inicio, 0, set(), {inicio: None})
def iddfs(tablero, movimientos):
    max_profundidad = len(tablero) * len(tablero[0])

    for limite in range(max_profundidad):
        print("\nIntentando con límite:", limite)
        resultado = ldfs(tablero, movimientos, limite)
        if resultado:
            return resultado

    return None


# ==============================
# HEURÍSTICA (para Voraz y A*)
# ==============================

def heuristica(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def voraz(tablero, movimientos):
    inicio = (0,0)
    meta = (len(tablero)-1, len(tablero[0])-1)

    frontera = []
    heapq.heappush(frontera, (heuristica(inicio, meta), inicio))

    padres = {inicio: None}
    visitados = set()

    while frontera:
        _, actual = heapq.heappop(frontera)
        print("Explorando:", actual)
        time.sleep(0.05)

        if actual == meta:
            return reconstruir_camino(padres, meta)

        visitados.add(actual)

        for dx, dy in movimientos:
            nx, ny = actual[0] + dx, actual[1] + dy
            vecino = (nx, ny)
            if es_valido(tablero, nx, ny) and vecino not in visitados:
                heapq.heappush(frontera, (heuristica(vecino, meta), vecino))
                padres[vecino] = actual

    return None




# ==============================
# A*
# ==============================

def a_star(tablero, movimientos):
    inicio = (0,0)
    meta = (len(tablero)-1, len(tablero[0])-1)

    frontera = []
    heapq.heappush(frontera, (0, inicio))

    padres = {inicio: None}
    costo = {inicio: 0}

    while frontera:
        _, actual = heapq.heappop(frontera)

        print("Explorando:", actual)
        time.sleep(0.1)

        if actual == meta:
            return reconstruir_camino(padres, meta)

        for dx, dy in movimientos:
            nx, ny = actual[0] + dx, actual[1] + dy
            if es_valido(tablero, nx, ny):
                vecino = (nx, ny)
                nuevo_costo = costo[actual] + 1

                if vecino not in costo or nuevo_costo < costo[vecino]:
                    costo[vecino] = nuevo_costo
                    prioridad = nuevo_costo + heuristica(vecino, meta)
                    heapq.heappush(frontera, (prioridad, vecino))
                    padres[vecino] = actual

    return None

def busqueda_tabu(tablero, movimientos, tam_tabu=10):
    inicio = (0,0)
    meta = (len(tablero)-1, len(tablero[0])-1)

    actual = inicio
    tabu = []
    padres = {inicio: None}

    while actual != meta:
        print("Explorando:", actual)
        time.sleep(0.05)

        vecinos = []
        for dx, dy in movimientos:
            nx, ny = actual[0] + dx, actual[1] + dy
            vecino = (nx, ny)
            if es_valido(tablero, nx, ny) and vecino not in tabu:
                vecinos.append(vecino)

        if not vecinos:
            return None

        vecino = min(vecinos, key=lambda x: heuristica(x, meta))
        padres[vecino] = actual

        tabu.append(actual)
        if len(tabu) > tam_tabu:
            tabu.pop(0)

        actual = vecino

    return reconstruir_camino(padres, meta)

import math

def recocido_simulado(tablero, movimientos):
    inicio = (0,0)
    meta = (len(tablero)-1, len(tablero[0])-1)

    actual = inicio
    padres = {inicio: None}

    temperatura = 100
    enfriamiento = 0.95

    while temperatura > 0.1:
        print("Explorando:", actual, "Temp:", round(temperatura,2))
        time.sleep(0.05)

        if actual == meta:
            return reconstruir_camino(padres, meta)

        vecinos = []
        for dx, dy in movimientos:
            nx, ny = actual[0] + dx, actual[1] + dy
            if es_valido(tablero, nx, ny):
                vecinos.append((nx, ny))

        if not vecinos:
            return None

        siguiente = random.choice(vecinos)

        delta = heuristica(siguiente, meta) - heuristica(actual, meta)

        if delta < 0 or random.random() < math.exp(-delta / temperatura):
            padres[siguiente] = actual
            actual = siguiente

        temperatura *= enfriamiento

    return None


# ==============================
# MENÚ PRINCIPAL
# ==============================

def menu():
    print("===== SIMULADOR DE LABERINTO =====\n")

    # Datos del tablero
    m = int(input("Filas (m): "))
    n = int(input("Columnas (n): "))
    porcentaje = float(input("% Obstáculos: "))

    tablero = generar_tablero(m, n, porcentaje)
    movimientos = obtener_movimientos()

    print("\nTablero generado:\n")
    mostrar_tablero(tablero)

    # Menú de algoritmos
    print("Seleccione algoritmo:")
    print("1 - BFS")
    print("2 - DFS")
    print("3 - A*")
    print("4 - LDFS")
    print("5 - IDDFS")
    print("6 - Voraz")
    print("7 - Búsqueda Tabú")
    print("8 - Recocido Simulado")

    opcion = int(input("Opción: "))

    print("\nIniciando búsqueda...\n")

    if opcion == 1:
        camino = bfs(tablero, movimientos)

    elif opcion == 2:
        camino = dfs(tablero, movimientos)

    elif opcion == 3:
        camino = a_star(tablero, movimientos)

    elif opcion == 4:
        limite = int(input("Límite de profundidad: "))
        camino = ldfs(tablero, movimientos, limite)

    elif opcion == 5:
        camino = iddfs(tablero, movimientos)

    elif opcion == 6:
        camino = voraz(tablero, movimientos)

    elif opcion == 7:
        camino = busqueda_tabu(tablero, movimientos)

    elif opcion == 8:
        camino = recocido_simulado(tablero, movimientos)

    else:
        print("Opción inválida.")
        return

    # Mostrar resultado
    if camino:
        print("\nCamino encontrado:\n")
        for x, y in camino:
            if tablero[x][y] == " ":
                tablero[x][y] = "."
        mostrar_tablero(tablero)
    else:
        print("\nNo se encontró solución.")

if __name__ == "__main__":
    menu()