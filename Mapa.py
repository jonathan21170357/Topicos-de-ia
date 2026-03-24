import googlemaps
import webbrowser
from collections import deque
import math
from heapq import heappush, heappop
import os
import polyline
import random

API_KEY = "AIzaSyAUbNrDCnRiCqyIJ1lbe7l_8sryb2V3WUo"
gmaps = googlemaps.Client(key=API_KEY)

# -------------------------------
# Distancia
# -------------------------------
def distancia(a, b):
    return math.sqrt((a['lat']-b['lat'])**2 + (a['lng']-b['lng'])**2)

# -------------------------------
# Obtener nodos
# -------------------------------
def obtener_nodos(steps):
    nodos = []
    for step in steps:
        puntos = polyline.decode(step['polyline']['points'])
        for lat, lng in puntos:
            nodos.append({'lat': lat, 'lng': lng})
    return nodos

# -------------------------------
# Grafo balanceado
# -------------------------------
def construir_grafo_real(nodos):
    G = {}

    for i in range(len(nodos)):
        G[i] = []

    for i in range(len(nodos)):
        if i+1 < len(nodos):
            G[i].append(i+1)
            G[i+1].append(i)

        if i+2 < len(nodos):
            G[i].append(i+2)
            G[i+2].append(i)

        if i+3 < len(nodos):
            G[i].append(i+3)
            G[i+3].append(i)

    return G

# -------------------------------
# BFS
# -------------------------------
def bfs(G, inicio, fin):
    cola = deque([[inicio]])
    visitados = set([inicio])

    while cola:
        ruta = cola.popleft()
        nodo = ruta[-1]

        if nodo == fin:
            return ruta

        vecinos = list(G[nodo])
        vecinos.reverse()

        for vecino in vecinos:
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(ruta + [vecino])

    return None

# -------------------------------
# DFS
# -------------------------------
def dfs(G, inicio, fin):
    pila = [[inicio]]
    visitados = set([inicio])

    while pila:
        ruta = pila.pop()
        nodo = ruta[-1]

        if nodo == fin:
            return ruta

        vecinos = list(G[nodo])
        random.shuffle(vecinos)

        for vecino in vecinos:
            if vecino not in visitados:
                visitados.add(vecino)
                pila.append(ruta + [vecino])

    return None

# -------------------------------
# A*
# -------------------------------
def astar(G, inicio, fin, nodos):
    open_set = []
    heappush(open_set, (0, inicio))

    came_from = {}
    g_score = {inicio: 0}

    while open_set:
        _, actual = heappop(open_set)

        if actual == fin:
            ruta = []
            while actual in came_from:
                ruta.append(actual)
                actual = came_from[actual]
            ruta.append(inicio)
            return ruta[::-1]

        for vecino in G[actual]:
            tentative_g = g_score[actual] + 1

            if vecino not in g_score or tentative_g < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = tentative_g

                h = distancia(nodos[vecino], nodos[fin])
                f = tentative_g + h

                heappush(open_set, (f, vecino))

    return None

# -------------------------------
# 🔥 Desplazar rutas para que no se encimen
# -------------------------------
def desplazar_ruta(ruta, nodos, offset):
    nueva = []
    for i in ruta:
        nueva.append({
            'lat': nodos[i]['lat'] + offset,
            'lng': nodos[i]['lng'] + offset
        })
    return nueva

# -------------------------------
# Generar mapa
# -------------------------------
def generar_mapa(rutas, nodos, origen, destino):
    colores = ["#0000FF", "#00AA00", "#FF0000"]
    offsets = [0.00005, 0, -0.00005]  # 🔥 separación visual

    centro = nodos[rutas[0][0]]

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparación Algoritmos</title>
        <script src="https://maps.googleapis.com/maps/api/js?key={API_KEY}"></script>
    </head>
    <body>
    <div id="map" style="height:100vh;"></div>

    <script>
    function initMap() {{
        var map = new google.maps.Map(document.getElementById('map'), {{
            zoom: 15,
            center: {{lat: {centro['lat']}, lng: {centro['lng']}}}
        }});
    """

    for i, ruta in enumerate(rutas):
        ruta_offset = desplazar_ruta(ruta, nodos, offsets[i])
        path = [f"{{lat:{p['lat']}, lng:{p['lng']}}}" for p in ruta_offset]

        html += f"""
        new google.maps.Polyline({{
            path: [{','.join(path)}],
            strokeColor: "{colores[i]}",
            strokeWeight: 5,
            map: map
        }});
        """

    html += """
    }
    window.onload = initMap;
    </script>
    </body>
    </html>
    """

    with open("comparacion.html", "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open("file://" + os.path.abspath("comparacion.html"))

    print("\n✅ Ahora se ven las 3 rutas correctamente")
    print("🔵 BFS | 🟢 DFS | 🔴 A*")

# -------------------------------
# MAIN
# -------------------------------
def main():
    origen = input("Dirección origen: ")
    destino = input("Dirección destino: ")

    directions = gmaps.directions(origen, destino, mode="driving")

    steps = directions[0]['legs'][0]['steps']

    nodos = obtener_nodos(steps)
    G = construir_grafo_real(nodos)

    inicio = 0
    fin = len(nodos) - 1

    ruta_bfs = bfs(G, inicio, fin)
    ruta_dfs = dfs(G, inicio, fin)
    ruta_astar = astar(G, inicio, fin, nodos)

    generar_mapa([ruta_bfs, ruta_dfs, ruta_astar], nodos, origen, destino)

if __name__ == "__main__":
    main()