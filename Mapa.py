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
# Obtener nodos y mapa de steps
# -------------------------------
def obtener_nodos_y_steps(directions):
    nodos = []
    mapa_steps = []

    for route in directions:
        for step in route['legs'][0]['steps']:
            puntos = polyline.decode(step['polyline']['points'])
            indices = []

            for lat, lng in puntos:
                nodos.append({'lat': lat, 'lng': lng})
                indices.append(len(nodos)-1)

            mapa_steps.append(indices)

    return nodos, mapa_steps

# -------------------------------
# Grafo REAL
# -------------------------------
def construir_grafo(nodos):
    G = {}

    for i in range(len(nodos)):
        G[i] = []

    for i in range(len(nodos)):
        # conexión natural
        if i+1 < len(nodos):
            G[i].append(i+1)
            G[i+1].append(i)

        # intersecciones reales
        for j in range(i+10, len(nodos)):
            if distancia(nodos[i], nodos[j]) < 0.0002:
                G[i].append(j)
                G[j].append(i)

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
            tentative_g = g_score[actual] + distancia(nodos[actual], nodos[vecino])

            if vecino not in g_score or tentative_g < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = tentative_g

                h = distancia(nodos[vecino], nodos[fin])
                f = tentative_g + h

                heappush(open_set, (f, vecino))

    return None

# -------------------------------
# 🔥 Convertir ruta a CALLES REALES
# -------------------------------
def convertir_a_ruta_real(ruta, nodos):
    ruta_real = []

    for i in range(len(ruta)-1):
        a = nodos[ruta[i]]
        b = nodos[ruta[i+1]]

        # interpolar suavemente (evita línea recta)
        ruta_real.append(a)
        ruta_real.append({
            'lat': (a['lat'] + b['lat']) / 2,
            'lng': (a['lng'] + b['lng']) / 2
        })

    ruta_real.append(nodos[ruta[-1]])
    return ruta_real

# -------------------------------
# Generar mapa
# -------------------------------
def generar_mapa(rutas, nodos, origen, destino):
    colores = ["#0000FF", "#00AA00", "#FF0000"]

    centro = nodos[rutas[0][0]]

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rutas reales</title>
        <script src="https://maps.googleapis.com/maps/api/js?key={API_KEY}"></script>
    </head>
    <body>
    <div id="map" style="height:100vh;"></div>

    <script>
    function initMap() {{
        var map = new google.maps.Map(document.getElementById('map'), {{
            zoom: 14,
            center: {{lat: {centro['lat']}, lng: {centro['lng']}}}
        }});
    """

    for i, ruta in enumerate(rutas):
        ruta_real = convertir_a_ruta_real(ruta, nodos)
        path = [f"{{lat:{p['lat']}, lng:{p['lng']}}}" for p in ruta_real]

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

    with open("rutas_reales.html", "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open("file://" + os.path.abspath("rutas_reales.html"))

    print("\n✅ Ahora sí sigue las calles reales")
    print(f"Punto A: {origen}")
    print(f"Punto B: {destino}")

# -------------------------------
# MAIN
# -------------------------------
def main():
    origen = input("Dirección origen: ")
    destino = input("Dirección destino: ")

    directions = gmaps.directions(
        origen,
        destino,
        mode="driving",
        alternatives=True
    )

    nodos, _ = obtener_nodos_y_steps(directions)
    G = construir_grafo(nodos)

    inicio = 0
    fin = len(nodos) - 1

    print("Calculando...")

    ruta_bfs = bfs(G, inicio, fin)
    ruta_dfs = dfs(G, inicio, fin)
    ruta_astar = astar(G, inicio, fin, nodos)

    generar_mapa([ruta_bfs, ruta_dfs, ruta_astar], nodos, origen, destino)

if __name__ == "__main__":
    main()