import googlemaps
import networkx as nx
import random
import math
from collections import deque
import folium

API_KEY = "AIzaSyAUbNrDCnRiCqyIJ1lbe7l_8sryb2V3WUo"
gmaps = googlemaps.Client(key=API_KEY)

# -------------------------------
# Crear grafo desde Google Maps
# -------------------------------
def obtener_ruta(origen, destino):
    print(f"Buscando ruta de: '{origen}' a '{destino}'")
    directions = gmaps.directions(origen, destino, mode="driving")
    pasos = directions[0]['legs'][0]['steps']

    grafo = nx.Graph()

    for step in pasos:
        start = (step['start_location']['lat'], step['start_location']['lng'])
        end = (step['end_location']['lat'], step['end_location']['lng'])
        distancia = step['distance']['value']
        grafo.add_edge(start, end, weight=distancia)

    return grafo

# -------------------------------
# Algoritmos de búsqueda (BFS, DFS, ILDFS, Voraz, A*, Tabú, Recocido)
# -------------------------------

def bfs(grafo, inicio, meta):
    visitados = set()
    cola = deque([(inicio,[inicio])])
    while cola:
        nodo, camino = cola.popleft()
        print("Visitando:", nodo)
        if nodo == meta:
            return camino
        visitados.add(nodo)
        for vecino in grafo.neighbors(nodo):
            if vecino not in visitados:
                cola.append((vecino, camino+[vecino]))
    return None

def dfs(grafo, nodo, meta, visitados=None, camino=None):
    if visitados is None:
        visitados=set()
        camino=[nodo]
    print("Visitando:", nodo)
    if nodo==meta:
        return camino
    visitados.add(nodo)
    for vecino in grafo.neighbors(nodo):
        if vecino not in visitados:
            res = dfs(grafo,vecino,meta,visitados,camino+[vecino])
            if res:
                return res
    return None

def dls(grafo, nodo, meta, limite, camino):
    print("Visitando:", nodo, "nivel:", limite)
    if nodo == meta:
        return camino
    if limite <= 0:
        return None
    for vecino in grafo.neighbors(nodo):
        res = dls(grafo, vecino, meta, limite-1, camino+[vecino])
        if res:
            return res
    return None

def ildfs(grafo, inicio, meta, max_depth=20):
    for depth in range(max_depth):
        print("Profundidad:",depth)
        res = dls(grafo, inicio, meta, depth, [inicio])
        if res:
            return res
    return None

def heuristica(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def voraz(grafo, inicio, meta):
    actual = inicio
    camino=[actual]
    visitados=set()
    while actual != meta:
        visitados.add(actual)
        vecinos=list(grafo.neighbors(actual))
        mejor=None
        mejor_h=999999
        for v in vecinos:
            if v not in visitados:
                h=heuristica(v,meta)
                if h < mejor_h:
                    mejor=v
                    mejor_h=h
        if mejor is None:
            return None
        print("Moviendo a:",mejor)
        camino.append(mejor)
        actual=mejor
    return camino

def astar(grafo,inicio,meta):
    abiertos=[(inicio,[inicio],0)]
    visitados=set()
    while abiertos:
        abiertos.sort(key=lambda x: x[2])
        nodo,camino,costo=abiertos.pop(0)
        print("Visitando:",nodo)
        if nodo==meta:
            return camino
        visitados.add(nodo)
        for vecino in grafo.neighbors(nodo):
            if vecino not in visitados:
                g=costo+grafo[nodo][vecino]['weight']
                h=heuristica(vecino,meta)
                f=g+h
                abiertos.append((vecino,camino+[vecino],f))
    return None

def tabu_search(grafo,inicio,meta,iteraciones=50):
    actual=inicio
    mejor=[inicio]
    tabu=set()
    for i in range(iteraciones):
        vecinos=list(grafo.neighbors(actual))
        vecinos=[v for v in vecinos if v not in tabu]
        if not vecinos:
            break
        siguiente=min(vecinos,key=lambda x:heuristica(x,meta))
        print("Iter",i,"->",siguiente)
        mejor.append(siguiente)
        tabu.add(actual)
        actual=siguiente
        if actual==meta:
            return mejor
    return mejor

def simulated_annealing(grafo,inicio,meta):
    actual=inicio
    camino=[inicio]
    T=100
    while T>1:
        vecinos=list(grafo.neighbors(actual))
        siguiente=random.choice(vecinos)
        delta=heuristica(siguiente,meta)-heuristica(actual,meta)
        if delta < 0 or random.random()<math.exp(-delta/T):
            actual=siguiente
            camino.append(actual)
            print("Moviendo a:",actual)
        if actual==meta:
            return camino
        T*=0.9
    return camino

# -------------------------------
# Generar archivo HTML con Folium
# -------------------------------
def generar_mapa(camino, nombre_archivo="mi_ruta.html"):
    if not camino or len(camino) < 2:
        print("No hay suficientes puntos para graficar.")
        return
    mapa = folium.Map(location=camino[0], zoom_start=15)
    folium.Marker(camino[0], popup="Inicio", icon=folium.Icon(color='green')).add_to(mapa)
    folium.Marker(camino[-1], popup="Meta", icon=folium.Icon(color='red')).add_to(mapa)
    folium.PolyLine(camino, color="blue", weight=5, opacity=0.8).add_to(mapa)
    mapa.save(nombre_archivo)
    print(f"\n✅ Mapa generado: {nombre_archivo}")

# -------------------------------
# Menu interactivo
# -------------------------------
def menu():
    origen=input("Origen: ")
    destino=input("Destino: ")
    grafo=obtener_ruta(origen,destino)
    nodos=list(grafo.nodes)
    inicio=nodos[0]
    meta=nodos[-1]

    print("""
1 BFS
2 DFS
3 ILDFS
4 Voraz
5 A*
6 Tabú
7 Recocido Simulado
""")
    opcion=int(input("Selecciona algoritmo: "))
    if opcion==1:
        camino=bfs(grafo,inicio,meta)
    elif opcion==2:
        camino=dfs(grafo,inicio,meta)
    elif opcion==3:
        camino=ildfs(grafo,inicio,meta)
    elif opcion==4:
        camino=voraz(grafo,inicio,meta)
    elif opcion==5:
        camino=astar(grafo,inicio,meta)
    elif opcion==6:
        camino=tabu_search(grafo,inicio,meta)
    elif opcion==7:
        camino=simulated_annealing(grafo,inicio,meta)
    else:
        print("Opción inválida")
        return

    print("\nRuta encontrada:")
    print(camino)
    if camino:
        generar_mapa(camino)
    else:
        print("El algoritmo no devolvió una ruta válida.")

menu()