import random
import math
import time
from collections import deque
import heapq

N = 8

def imprimir_tablero(tablero):
    for i in range(N):
        fila = ""
        for j in range(N):
            if tablero[j] == i:
                fila += "Q "
            else:
                fila += ". "
        print(fila)
    print("\n")

def conflictos(tablero):

    ataques = 0
    n = len(tablero)

    for i in range(n):
        for j in range(i+1, n):

            if tablero[i] == tablero[j]:
                ataques += 1

            if abs(tablero[i] - tablero[j]) == abs(i - j):
                ataques += 1

    return ataques


def es_solucion(tablero):
    return conflictos(tablero) == 0
# ---------------- DFS ----------------

def dfs():
    stack = [([],0)]

    while stack:
        estado, col = stack.pop()

        if col == N:
            imprimir_tablero(estado)
            return estado

        for fila in range(N):
            nuevo = estado + [fila]
            if conflictos(nuevo) == 0:
                print("Paso DFS")
                imprimir_tablero(nuevo + [-1]*(N-len(nuevo)))
                stack.append((nuevo,col+1))


# ---------------- BFS ----------------

def bfs():
    cola = deque()
    cola.append(([],0))

    while cola:
        estado,col = cola.popleft()

        if col == N:
            imprimir_tablero(estado)
            return estado

        for fila in range(N):
            nuevo = estado + [fila]

            if conflictos(nuevo) == 0:
                print("Paso BFS")
                imprimir_tablero(nuevo + [-1]*(N-len(nuevo)))
                cola.append((nuevo,col+1))


# ---------------- LDFS / ILDFS ----------------

def dfs_limit(estado, col, limite):

    if col > limite:
        return None

    if col == N:
        imprimir_tablero(estado)
        return estado

    for fila in range(N):

        nuevo = estado + [fila]

        if conflictos(nuevo) == 0:

            print("Paso LDFS")
            imprimir_tablero(nuevo + [-1]*(N-len(nuevo)))

            resultado = dfs_limit(nuevo, col+1, limite)

            if resultado:
                return resultado

    return None

def ildfs():
    for limite in range(1, N+1):
        resultado = dfs_limit([],0,limite)
        if resultado:
            return resultado

# ---------------- Greedy ----------------

def greedy():

    tablero = [random.randint(0, N-1) for _ in range(N)]

    while True:

        print("Estado actual:")
        imprimir_tablero(tablero)

        conf_actual = conflictos(tablero)

        if conf_actual == 0:
            print("Solución encontrada")
            return tablero

        mejor_tablero = tablero
        mejor_conf = conf_actual

        for col in range(N):
            for fila in range(N):

                if fila != tablero[col]:

                    nuevo = tablero.copy()
                    nuevo[col] = fila

                    conf = conflictos(nuevo)

                    if conf < mejor_conf:
                        mejor_conf = conf
                        mejor_tablero = nuevo

        if mejor_tablero == tablero:
            # mínimo local → reinicio aleatorio
            tablero = [random.randint(0, N-1) for _ in range(N)]
        else:
            tablero = mejor_tablero

# ---------------- A* ----------------

def astar():

    cola = []
    estado = []

    heapq.heappush(cola,(0,estado))

    while cola:

        _,estado = heapq.heappop(cola)
        col = len(estado)

        if col == N:
            imprimir_tablero(estado)
            return estado

        for fila in range(N):

            nuevo = estado + [fila]

            if conflictos(nuevo) == 0:

                g = len(nuevo)
                h = conflictos(nuevo)
                f = g + h

                print("Paso A*")
                imprimir_tablero(nuevo + [-1]*(N-len(nuevo)))

                heapq.heappush(cola,(f,nuevo))


# ---------------- Tabu Search ----------------

def tabu_search(iteraciones=1000):

    tablero = [random.randint(0,N-1) for _ in range(N)]
    tabu = []

    for _ in range(iteraciones):

        imprimir_tablero(tablero)

        if es_solucion(tablero):
            return tablero

        vecinos = []

        for col in range(N):
            for fila in range(N):

                nuevo = tablero.copy()
                nuevo[col] = fila

                if nuevo not in tabu:
                    vecinos.append((conflictos(nuevo),nuevo))

        vecinos.sort()

        mejor = vecinos[0][1]

        tabu.append(tablero)

        if len(tabu) > 50:
            tabu.pop(0)

        tablero = mejor


# ---------------- Simulated Annealing ----------------

def simulated_annealing():

    tablero = [random.randint(0,N-1) for _ in range(N)]

    T = 100
    enfriamiento = 0.95

    while T > 0.1:

        imprimir_tablero(tablero)

        if es_solucion(tablero):
            print("Solución encontrada")
            return tablero

        col = random.randint(0,N-1)
        fila = random.randint(0,N-1)

        nuevo = tablero.copy()
        nuevo[col] = fila

        delta = conflictos(nuevo) - conflictos(tablero)

        if delta < 0 or random.random() < math.exp(-delta/T):
            tablero = nuevo

        T *= enfriamiento

    # Verificación final
    if es_solucion(tablero):
        print("Solución encontrada")
    else:
        print("No encontró solución perfecta, conflictos:", conflictos(tablero))

    imprimir_tablero(tablero)

    return tablero


# ---------------- MENU ----------------

def menu():

    while True:

        print("\nPROBLEMA DE LAS N REINAS\n")

        print("1. BFS")
        print("2. DFS")
        print("3. LDFS / ILDFS")
        print("4. Búsqueda Voraz")
        print("5. A*")
        print("6. Búsqueda Tabú")
        print("7. Recocido Simulado")
        print("8. Salir")

        op = input("Seleccione algoritmo: ")

        if op == "1":
            bfs()

        elif op == "2":
            dfs()

        elif op == "3":
            ildfs()

        elif op == "4":
            greedy()

        elif op == "5":
            astar()

        elif op == "6":
            tabu_search()

        elif op == "7":
            simulated_annealing()

        elif op == "8":
            break


menu()