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
# Obtener nodos y conexiones reales
# -------------------------------
def obtener_nodos(directions):
    nodos = []
    conexiones = []

    for route in directions:
        prev_index = None

        for step in route['legs'][0]['steps']:
            puntos = polyline.decode(step['polyline']['points'])

            for lat, lng in puntos:
                nodos.append({'lat': lat, 'lng': lng})
                idx = len(nodos) - 1

                if prev_index is not None:
                    conexiones.append((prev_index, idx))

                prev_index = idx

    return nodos, conexiones

# -------------------------------
# Construir grafo con intersecciones
# -------------------------------
def construir_grafo(nodos, conexiones):
    G = {i: [] for i in range(len(nodos))}

    # conexiones reales
    for a, b in conexiones:
        G[a].append(b)
        G[b].append(a)

    # 🔥 intersecciones controladas
    for i in range(len(nodos)):
        conexiones_agregadas = 0
        for j in range(i+10, len(nodos)):
            d = distancia(nodos[i], nodos[j])

            if 0.00005 < d < 0.0003:
                G[i].append(j)
                G[j].append(i)

                conexiones_agregadas += 1
                if conexiones_agregadas >= 3:
                    break

    return G

# -------------------------------
# BFS (modificado)
# -------------------------------
def bfs(G, inicio, fin):
    cola = deque([[inicio]])
    visitados = set([inicio])
    LIMITE = 200

    while cola:
        ruta = cola.popleft()
        nodo = ruta[-1]

        if nodo == fin:
            return ruta

        if len(ruta) > LIMITE:
            continue

        vecinos = list(G[nodo])
        random.shuffle(vecinos)

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
    LIMITE = 200

    while pila:
        ruta = pila.pop()
        nodo = ruta[-1]

        if nodo == fin:
            return ruta

        if len(ruta) > LIMITE:
            continue

        vecinos = list(G[nodo])
        random.shuffle(vecinos)

        for vecino in vecinos:
            if vecino not in ruta:
                pila.append(ruta + [vecino])

    return None

# -------------------------------
# A* (con penalización)
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
            tentative_g = g_score[actual] + 1 + random.uniform(0.2, 0.8)

            if vecino not in g_score or tentative_g < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = tentative_g

                h = distancia(nodos[vecino], nodos[fin])
                f = tentative_g + h

                heappush(open_set, (f, vecino))

    return None

# -------------------------------
# Generar mapa
# -------------------------------
def generar_mapa(rutas, nodos, origen, destino):
    colores = ["#0000FF", "#00AA00", "#FF0000"]
    offsets = [0.00005, 0, -0.00005]

    ruta_valida = next((r for r in rutas if r), None)

    if not ruta_valida:
        print("❌ No se encontró ninguna ruta")
        return

    centro = nodos[ruta_valida[0]]

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rutas</title>
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
        if not ruta:
            continue

        path = [
            f"{{lat:{nodos[j]['lat'] + offsets[i]}, lng:{nodos[j]['lng'] + offsets[i]}}}"
            for j in ruta
        ]

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

    with open("mapa_final.html", "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open("file://" + os.path.abspath("mapa_final.html"))

    print("\n✅ MAPA GENERADO")
    print(f"Punto A: {origen}")
    print(f"Punto B: {destino}")
    print("🔵 BFS | 🟢 DFS | 🔴 A*")

# -------------------------------
# MAIN
# -------------------------------
def main():
    origen = input("Origen: ")
    destino = input("Destino: ")

    directions = gmaps.directions(
        origen,
        destino,
        mode="driving",
        alternatives=True
    )

    nodos, conexiones = obtener_nodos(directions)
    G = construir_grafo(nodos, conexiones)

    inicio = 0
    fin = len(nodos) - 1

    print("Calculando...")

    ruta_bfs = bfs(G, inicio, fin)
    ruta_dfs = dfs(G, inicio, fin)
    ruta_astar = astar(G, inicio, fin, nodos)

    print("BFS:", "OK" if ruta_bfs else "No ruta")
    print("DFS:", "OK" if ruta_dfs else "No ruta")
    print("A* :", "OK" if ruta_astar else "No ruta")

    generar_mapa([ruta_bfs, ruta_dfs, ruta_astar], nodos, origen, destino)

# -------------------------------
if __name__ == "__main__":
    main()