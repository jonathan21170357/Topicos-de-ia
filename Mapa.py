import googlemaps
import networkx as nx
import random
import math
import folium
import time
import osmnx as ox
from collections import deque

API_KEY = "AIzaSyAUbNrDCnRiCqyIJ1lbe7l_8sryb2V3WUo" 
gmaps = googlemaps.Client(key=API_KEY)

# -------------------------------
# LOG (consola + archivo)
# -------------------------------
def escribir(log, texto):
    print(texto)
    log.write(texto + "\n")

# -------------------------------
# Geocodificar
# -------------------------------
def obtener_coordenadas(lugar):
    geo = gmaps.geocode(lugar)
    lat = geo[0]['geometry']['location']['lat']
    lng = geo[0]['geometry']['location']['lng']
    return (lat, lng)

# -------------------------------
# Grafo de Calles Reales (OSMnx)
# -------------------------------
def crear_grafo_real(origen, destino):
    print("Descargando mapa de calles reales... Esto puede tardar un momento dependiendo de la distancia.")
    # Punto medio para descargar el mapa
    clat = (origen[0] + destino[0]) / 2
    clng = (origen[1] + destino[1]) / 2
    
    # Distancia aproximada en metros + un margen (buffer) de 2km
    dist_grados = math.sqrt((origen[0]-destino[0])**2 + (origen[1]-destino[1])**2)
    dist_metros = dist_grados * 111320 + 2000 
    
    # Descargar grafo de red de calles para conducir
    G_osm = ox.graph_from_point((clat, clng), dist=dist_metros, network_type='drive')
    
    # Convertir a un grafo no dirigido simple para tus algoritmos
    G = nx.Graph()
    for n, data in G_osm.nodes(data=True):
        G.add_node(n, y=data['y'], x=data['x'])
        
    for u, v, data in G_osm.edges(data=True):
        # Tomamos la longitud de la calle como peso, normalizada a grados aprox.
        peso = data.get('length', 10) / 111320 
        G.add_edge(u, v, weight=peso)
        
    return G

def nodo_mas_cercano(G, lat, lng):
    # Encuentra el nodo (intersección) más cercano a unas coordenadas dadas
    mejor_nodo = None
    min_dist = float('inf')
    for n, data in G.nodes(data=True):
        d = math.sqrt((data['y']-lat)**2 + (data['x']-lng)**2)
        if d < min_dist:
            min_dist = d
            mejor_nodo = n
    return mejor_nodo

# -------------------------------
# Heurística Adaptada
# -------------------------------
def h(G, a, b):
    # Ahora lee las coordenadas directamente de los atributos del nodo en el grafo real
    lat1, lng1 = G.nodes[a]['y'], G.nodes[a]['x']
    lat2, lng2 = G.nodes[b]['y'], G.nodes[b]['x']
    return math.sqrt((lat1-lat2)**2 + (lng1-lng2)**2)

# -------------------------------
# ALGORITMOS (Adaptados para pasar G a la heurística)
# -------------------------------
def bfs(G, inicio, meta, log):
    t=time.time()
    visitados=[]
    cola=deque([[inicio]])
    paso=0
    
    while cola:
        paso+=1
        camino=cola.popleft()
        nodo=camino[-1]
        
        if nodo not in visitados:
            visitados.append(nodo)
            if nodo==meta:
                escribir(log, f"🎯 Meta encontrada en paso {paso}")
                return camino,visitados,time.time()-t
            
            for v in G.neighbors(nodo):
                cola.append(camino+[v])
    return None,visitados,time.time()-t

def dfs(G,inicio,meta,log):
    t=time.time()
    visitados=[]
    pila=[[inicio]]
    paso=0
    
    while pila:
        paso+=1
        camino=pila.pop()
        nodo=camino[-1]
        
        if nodo not in visitados:
            visitados.append(nodo)
            if nodo==meta:
                escribir(log, f"🎯 Meta encontrada en paso {paso}")
                return camino,visitados,time.time()-t
            
            for v in G.neighbors(nodo):
                pila.append(camino+[v])
    return None,visitados,time.time()-t

def astar(G,inicio,meta,log):
    t=time.time()
    abiertos=[(inicio,[inicio],0)]
    visitados=[]
    paso=0
    
    while abiertos:
        paso+=1
        abiertos.sort(key=lambda x:x[2])
        nodo,camino,costo=abiertos.pop(0)
        
        if nodo not in visitados:
            visitados.append(nodo)
            if nodo==meta:
                escribir(log, f"🎯 Meta encontrada en paso {paso}")
                return camino,visitados,time.time()-t
            
            for v in G.neighbors(nodo):
                g = costo + G[nodo][v].get('weight', h(G, nodo, v))
                f = g + h(G, v, meta)
                abiertos.append((v, camino+[v], f))
    
    return None,visitados,time.time()-t

# -------------------------------
# VORAZ (Greedy Best-First Search)
# -------------------------------
def voraz(G, inicio, meta, log):
    t = time.time()
    # La lista 'abiertos' guarda tuplas: (nodo, camino_recorrido, distancia_estimada_a_meta)
    abiertos = [(inicio, [inicio], h(G, inicio, meta))]
    visitados = []
    paso = 0
    
    while abiertos:
        paso += 1
        
        # Ordenamos la lista basándonos SOLO en la heurística h (el índice 2 de la tupla)
        # Esto es lo que lo hace "Voraz": siempre elige lo que parece más cerca de la meta.
        abiertos.sort(key=lambda x: x[2])
        
        # Extraemos el nodo que parece más prometedor
        nodo, camino, heuristica = abiertos.pop(0)
        
        if nodo not in visitados:
            visitados.append(nodo)
            
            # Si llegamos a la meta, terminamos
            if nodo == meta:
                escribir(log, f"🎯 Meta encontrada por Voraz en paso {paso}")
                return camino, visitados, time.time() - t
            
            # Si no es la meta, exploramos sus vecinos
            for v in G.neighbors(nodo):
                if v not in visitados:
                    # Calculamos qué tan lejos está el vecino de la meta
                    distancia_a_meta = h(G, v, meta)
                    abiertos.append((v, camino + [v], distancia_a_meta))
    
    # Si se agotan las opciones sin llegar a la meta
    return None, visitados, time.time() - t

# (Puedes re-implementar ILDFS, Tabú y Recocido aquí siguiendo el mismo patrón de cambiar h(a,b) por h(G,a,b))

# -------------------------------
# COSTO
# -------------------------------
def costo(G, camino):
    if not camino: return 0
    return sum(h(G, camino[i], camino[i+1]) for i in range(len(camino)-1))

# -------------------------------
# MAPA
# -------------------------------
# -------------------------------
# MAPA (Restaurado a tu diseño original)
# -------------------------------
# -------------------------------
# MAPA (Sin garabatos ni saltos)
# -------------------------------
def mapa(G, origen, destino, visitados, camino, color):
    # Centramos el mapa en las coordenadas de origen
    m = folium.Map(location=origen, zoom_start=14)

    # 🔵 PROCESO (área explorada respetando las calles reales)
    # Creamos un "subgrafo" que contiene solo los nodos que el algoritmo visitó.
    # .edges() nos da exactamente las calles físicas que conectan esos nodos.
    calles_exploradas = G.subgraph(visitados).edges()
    
    for u, v in calles_exploradas:
        coord_u = (G.nodes[u]['y'], G.nodes[u]['x'])
        coord_v = (G.nodes[v]['y'], G.nodes[v]['x'])
        
        folium.PolyLine(
            [coord_u, coord_v],
            color='lightblue',
            weight=2,
            opacity=0.6
        ).add_to(m)

    # 📍 Inicio
    folium.Marker(
        origen,
        popup="Inicio",
        icon=folium.Icon(color='green')
    ).add_to(m)

    # 📍 Meta
    folium.Marker(
        destino,
        popup="Meta",
        icon=folium.Icon(color='red')
    ).add_to(m)

    # 🟢 CAMINO FINAL (resaltado)
    if camino:
        coords_camino = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in camino]
        folium.PolyLine(
            coords_camino,
            color=color,
            weight=6,
            opacity=0.9
        ).add_to(m)

    # Guardamos el archivo HTML
    m.save("mapa_final.html")
# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    log = open("proceso_busqueda.txt", "w", encoding="utf-8")

    origen_txt=input("Origen (ej. 'Zocalo, CDMX'): ")
    destino_txt=input("Destino (ej. 'Bellas Artes, CDMX'): ")

    # Asegúrate de mantener búsquedas dentro de la misma ciudad para evitar tiempos de descarga excesivos
    origen=obtener_coordenadas(origen_txt)
    destino=obtener_coordenadas(destino_txt)

    # 1. Crear el grafo basado en calles reales
    G = crear_grafo_real(origen, destino)

    # 2. Mapear origen y destino al nodo (intersección) más cercano en la vida real
    nodo_inicio = nodo_mas_cercano(G, origen[0], origen[1])
    nodo_meta = nodo_mas_cercano(G, destino[0], destino[1])

    print("""
    1 BFS
    2 DFS
    3 Voraz
    4 A*
    (He acortado la lista en el ejemplo, pero puedes añadir los demás)
    """)

    op=int(input("Selecciona: "))

    algos=[bfs,dfs,voraz,astar]
    nombres=["BFS","DFS","VORAZ","A*"]
    colores={"BFS":"blue", "DFS":"purple", "VORAZ":"red", "A*":"green"}

    print(f"Buscando ruta con {nombres[op-1]}...")
    camino,visitados,tiempo=algos[op-1](G, nodo_inicio, nodo_meta, log)

    # MÉTRICAS
    escribir(log, "\n--- MÉTRICAS ---")
    escribir(log, f"Algoritmo: {nombres[op-1]}")
    escribir(log, f"Visitados: {len(visitados)}")
    escribir(log, f"Longitud (nodos): {len(camino) if camino else 0}")
    escribir(log, f"Costo: {costo(G, camino)}")
    escribir(log, f"Tiempo: {tiempo:.4f} segundos")

    # Mapeo usando las coordenadas originales pero la ruta sobre la red de calles
    mapa(G, origen, destino, visitados, camino, colores.get(nombres[op-1], "black"))

    log.close()

    print("\n✅ mapa_final.html generado (Ahora usa calles reales)")
    print("✅ proceso_busqueda.txt generado")