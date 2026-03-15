import random
import math
import heapq
import time
import os
from collections import deque

N = 8
nodos_explorados = 0


# -------- COLORES --------
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
ROJO = "\033[91m"
VERDE = "\033[92m"
AMARILLO = "\033[93m"
RESET = "\033[0m"


# -------- LIMPIAR CONSOLA --------

def limpiar():
    os.system('cls' if os.name == 'nt' else 'clear')


# -------- TABLERO --------

def mostrar_tablero(tablero, explorando=None, aceptado=None):

    limpiar()

    for fila in range(N):

        linea = ""

        for col in range(N):

            if col < len(tablero) and tablero[col] == fila:
                linea += ROJO + "Q " + RESET

            elif aceptado and aceptado == (fila, col):
                linea += VERDE + "* " + RESET

            elif explorando and explorando == (fila, col):
                linea += AMARILLO + "* " + RESET

            else:
                linea += ". "

        print(linea)

    time.sleep(0.01)


# -------- CONFLICTOS --------

def conflictos(tablero):

    ataques = 0

    for i in range(len(tablero)):
        for j in range(i + 1, len(tablero)):

            if tablero[i] == tablero[j]:
                ataques += 1

            if abs(tablero[i] - tablero[j]) == abs(i - j):
                ataques += 1

    return ataques


def es_solucion(tablero):
    return len(tablero) == N and conflictos(tablero) == 0


# -------- RESULTADO --------

def verificar(tablero, tiempo):

    limpiar()

    if es_solucion(tablero):
        print("Solución encontrada\n")
    else:
        print("Solución más cercana encontrada")
        print("No se pudo encontrar solución correcta al 100%\n")

    mostrar_tablero(tablero)

    print("Conflictos:", conflictos(tablero))
    print("Nodos explorados:", nodos_explorados)
    print("Tiempo:", round(tiempo,4),"segundos")


# -------- DFS --------

def dfs():

    global nodos_explorados

    stack=[([],0)]
    mejor=[]

    while stack:

        estado,col=stack.pop()

        if len(estado)>len(mejor):
            mejor=estado

        if col==N:
            return estado

        for fila in range(N):

            nodos_explorados+=1

            mostrar_tablero(estado,(fila,col))

            nuevo=estado+[fila]

            if conflictos(nuevo)==0:

                mostrar_tablero(nuevo,aceptado=(fila,col))

                stack.append((nuevo,col+1))

    return mejor


# -------- BFS --------

def bfs():

    global nodos_explorados

    cola=deque()
    cola.append(([],0))
    mejor=[]

    while cola:

        estado,col=cola.popleft()

        if len(estado)>len(mejor):
            mejor=estado

        if col==N:
            return estado

        for fila in range(N):

            nodos_explorados+=1

            mostrar_tablero(estado,(fila,col))

            nuevo=estado+[fila]

            if conflictos(nuevo)==0:

                mostrar_tablero(nuevo,aceptado=(fila,col))

                cola.append((nuevo,col+1))

    return mejor


# -------- LDFS --------

def dfs_limit(estado,col,limite,mejor):

    global nodos_explorados

    if col>limite:
        return None

    if len(estado)>len(mejor[0]):
        mejor[0]=estado

    if col==N:
        return estado

    for fila in range(N):

        nodos_explorados+=1

        mostrar_tablero(estado,(fila,col))

        nuevo=estado+[fila]

        if conflictos(nuevo)==0:

            resultado=dfs_limit(nuevo,col+1,limite,mejor)

            if resultado:
                return resultado

    return None


def ildfs():

    mejor=[[]]

    for limite in range(1,N+1):

        resultado=dfs_limit([],0,limite,mejor)

        if resultado:
            return resultado

    return mejor[0]


# -------- GREEDY --------

def greedy():

    global nodos_explorados

    tablero=[random.randint(0,N-1) for _ in range(N)]
    mejor=tablero

    for _ in range(200):

        mostrar_tablero(tablero)

        if conflictos(tablero)==0:
            return tablero

        mejor_tablero=tablero
        mejor_conf=conflictos(tablero)

        for col in range(N):
            for fila in range(N):

                nodos_explorados+=1

                mostrar_tablero(tablero,(fila,col))

                nuevo=tablero.copy()
                nuevo[col]=fila

                conf=conflictos(nuevo)

                if conf<mejor_conf:
                    mejor_conf=conf
                    mejor_tablero=nuevo

        tablero=mejor_tablero

        if conflictos(tablero)<conflictos(mejor):
            mejor=tablero

    return mejor


# -------- A* --------

def astar():

    global nodos_explorados

    cola=[]
    heapq.heappush(cola,(0,[]))
    mejor=[]

    while cola:

        _,estado=heapq.heappop(cola)
        col=len(estado)

        if len(estado)>len(mejor):
            mejor=estado

        if col==N:
            return estado

        for fila in range(N):

            nodos_explorados+=1

            mostrar_tablero(estado,(fila,col))

            nuevo=estado+[fila]

            if conflictos(nuevo)==0:

                g=len(nuevo)
                h=conflictos(nuevo)
                f=g+h

                mostrar_tablero(nuevo,aceptado=(fila,col))

                heapq.heappush(cola,(f,nuevo))

    return mejor


# -------- TABU SEARCH --------

def tabu_search(iteraciones=500):

    global nodos_explorados

    tablero=[random.randint(0,N-1) for _ in range(N)]
    mejor=tablero
    tabu=[]

    for _ in range(iteraciones):

        mostrar_tablero(tablero)

        if es_solucion(tablero):
            return tablero

        vecinos=[]

        for col in range(N):
            for fila in range(N):

                nodos_explorados+=1

                mostrar_tablero(tablero,(fila,col))

                if fila!=tablero[col]:

                    nuevo=tablero.copy()
                    nuevo[col]=fila

                    if nuevo not in tabu:
                        vecinos.append((conflictos(nuevo),nuevo))

        vecinos.sort()

        mejor_vecino=vecinos[0][1]

        if conflictos(mejor_vecino)<conflictos(mejor):
            mejor=mejor_vecino

        tabu.append(tablero)

        if len(tabu)>20:
            tabu.pop(0)

        tablero=mejor_vecino

    return mejor


# -------- SIMULATED ANNEALING --------

def simulated_annealing():

    global nodos_explorados

    tablero=[random.randint(0,N-1) for _ in range(N)]
    mejor=tablero

    T=50
    enfriamiento=0.95

    while T>0.1:

        mostrar_tablero(tablero)

        if es_solucion(tablero):
            return tablero

        col=random.randint(0,N-1)
        fila=random.randint(0,N-1)

        nodos_explorados+=1

        mostrar_tablero(tablero,(fila,col))

        nuevo=tablero.copy()
        nuevo[col]=fila

        delta=conflictos(nuevo)-conflictos(tablero)

        if delta<0 or random.random()<math.exp(-delta/T):
            tablero=nuevo

        if conflictos(tablero)<conflictos(mejor):
            mejor=tablero

        T*=enfriamiento

    return mejor


# -------- EJECUTAR --------

def ejecutar(algoritmo):

    global nodos_explorados

    nodos_explorados=0

    inicio=time.time()

    resultado=algoritmo()

    fin=time.time()

    verificar(resultado,fin-inicio)


# -------- MENU --------

def menu():

    while True:

        print("\nPROBLEMA DE LAS 4 REINAS\n")

        print("1 BFS")
        print("2 DFS")
        print("3 LDFS / IDDFS")
        print("4 Voraz")
        print("5 A*")
        print("6 Tabu Search")
        print("7 Recocido Simulado")
        print("8 Salir")

        op=input("Seleccione algoritmo: ")

        if op=="1":
            ejecutar(bfs)

        elif op=="2":
            ejecutar(dfs)

        elif op=="3":
            ejecutar(ildfs)

        elif op=="4":
            ejecutar(greedy)

        elif op=="5":
            ejecutar(astar)

        elif op=="6":
            ejecutar(tabu_search)

        elif op=="7":
            ejecutar(simulated_annealing)

        elif op=="8":
            break


menu()