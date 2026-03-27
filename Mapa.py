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
def distancia(a, b):
    return math.sqrt((a['lat']-b['lat'])**2 + (a['lng']-b['lng'])**2)

# -------------------------------
# -------------------------------
def obtener_nodos(directions):
    nodos = []
    conexiones = []

    # Ahora iteramos sobre TODAS las rutas que devuelva Google Maps, no solo la [0]
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
def construir_grafo(nodos, conexiones):
    G = {i: set() for i in range(len(nodos))}

    for a, b in conexiones:
        G[a].add(b)
        G[b].add(a)

  
    for i in range(len(nodos)):
        for j in range(i+3, len(nodos)):
            if distancia(nodos[i], nodos[j]) < 0.0002: 
                G[i].add(j)
                G[j].add(i)

    return {k: list(v) for k, v in G.items()}

# -------------------------------
# BFS
# -------------------------------
def bfs(G, inicio, fin, nodos):
    cola = deque([[inicio]])
    visitados = set([inicio])

    while cola:
        ruta = cola.popleft()
        nodo = ruta[-1]

        # MODIFICADO: Llegamos si estamos geográficamente cerca del destino
        if distancia(nodos[nodo], nodos[fin]) < 0.0005: 
            return ruta

        for vecino in G[nodo]:
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(ruta + [vecino])

    return None

# -------------------------------
# DFS
# -------------------------------
def dfs(G, inicio, fin, nodos, evitar):
    pila = [[inicio]]
    visitados = set()

    while pila:
        ruta = pila.pop()
        nodo = ruta[-1]

        # MODIFICADO
        if distancia(nodos[nodo], nodos[fin]) < 0.0005:
            return ruta

        if nodo not in visitados:
            visitados.add(nodo)
            vecinos = list(G[nodo])
            
            vecinos.sort(key=lambda x: 1 if x in evitar else 0, reverse=True)

            for vecino in vecinos:
                if vecino not in visitados:
                    pila.append(ruta + [vecino])

    return None

# -------------------------------
# A*
# -------------------------------
def astar(G, inicio, fin, nodos, evitar):
    open_set = []
    heappush(open_set, (0, inicio))

    came_from = {}
    g_score = {inicio: 0}

    while open_set:
        _, actual = heappop(open_set)

        # MODIFICADO
        if distancia(nodos[actual], nodos[fin]) < 0.0005:
            ruta = []
            while actual in came_from:
                ruta.append(actual)
                actual = came_from[actual]
            ruta.append(inicio)
            return ruta[::-1]

        for vecino in G[actual]:
            penal = 0.02 if vecino in evitar else 0
            tentative_g = g_score[actual] + distancia(nodos[actual], nodos[vecino]) + penal

            if vecino not in g_score or tentative_g < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = tentative_g

                h = distancia(nodos[vecino], nodos[fin])
                f = tentative_g + h

                heappush(open_set, (f, vecino))

    return None

# -------------------------------
# Voraz
# -------------------------------
def voraz(G, inicio, fin, nodos):
    actual = inicio
    ruta = [actual]
    visitados = set([inicio])

    while distancia(nodos[actual], nodos[fin]) >= 0.0005: 
        vecinos = [v for v in G[actual] if v not in visitados]
        
        if not vecinos:
            if len(ruta) > 1:
                ruta.pop()
                actual = ruta[-1]
                continue
            else:
                return None 

        siguiente = min(
            vecinos,
            key=lambda x: distancia(nodos[x], nodos[fin])
        )

        ruta.append(siguiente)
        visitados.add(siguiente)
        actual = siguiente

    return ruta

# -------------------------------
# Tabú
# -------------------------------
def tabu(G, inicio, fin, nodos):
    actual = inicio
    ruta = [actual]
    tabu_lista = set([inicio])

    while distancia(nodos[actual], nodos[fin]) >= 0.0005:
        vecinos = [v for v in G[actual] if v not in tabu_lista]

        if not vecinos:
            if len(ruta) > 1:
                ruta.pop()
                actual = ruta[-1]
                continue
            else:
                return None

        vecinos.sort(key=lambda x: distancia(nodos[x], nodos[fin]))
        siguiente = random.choice(vecinos[:3]) if len(vecinos) >= 3 else vecinos[0]

        ruta.append(siguiente)
        tabu_lista.add(siguiente)
        actual = siguiente

    return ruta

# -------------------------------
# Recocido Simulado
# -------------------------------
def recocido(G, inicio, fin, nodos):
    actual = inicio
    ruta = [actual]
    visitados = set([inicio])
    T = 1.0

    while distancia(nodos[actual], nodos[fin]) >= 0.0005:
        vecinos = [v for v in G[actual] if v not in visitados]
        
        if not vecinos:
            if len(ruta) > 1:
                ruta.pop()
                actual = ruta[-1]
                continue
            else:
                return None

        siguiente = random.choice(vecinos)
        delta = distancia(nodos[siguiente], nodos[fin]) - distancia(nodos[actual], nodos[fin])

        if delta < 0 or random.random() < math.exp(-delta / max(T, 0.001)):
            actual = siguiente
            ruta.append(actual)
            visitados.add(actual)

        T *= 0.95 

    return ruta

# -------------------------------
# -------------------------------
def generar_mapa(rutas, nodos):
    colores = ["#0000FF", "#008000", "#FF0000", "#FFA500", "#800080", "#00FFFF"]

    offsets = [0.00015, 0.00010, 0, -0.00010, -0.00015, 0.00020]

    ruta_valida = next((r for r in rutas if r), None)

    if not ruta_valida:
        print("❌ No se encontró ninguna ruta")
        return

    centro = nodos[ruta_valida[0]]

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
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

    nombres = ["BFS", "DFS", "A*", "Voraz", "Tabú", "Recocido"]

    for i, ruta in enumerate(rutas):
        if not ruta:
            print(f"⚠ El algoritmo {nombres[i]} no encontró un camino válido.")
            continue

        path = [
            f"{{lat:{nodos[j]['lat'] + offsets[i]}, lng:{nodos[j]['lng'] + offsets[i]}}}"
            for j in ruta
        ]

        html += f"""
        new google.maps.Polyline({{
            path: [{','.join(path)}],
            strokeColor: "{colores[i]}",
            strokeOpacity: 0.8,
            strokeWeight: 4,
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

    print("\n✅ MAPA GENERADO CORRECTAMENTE")
    print("✔ Todos los algoritmos ejecutados\n")
    
    # --- NUEVO: Leyenda de colores en consola ---
    print("--- LEYENDA DE COLORES ---")
    print("🔵 Azul    -> BFS")
    print("🟢 Verde   -> DFS")
    print("🔴 Rojo    -> A*")
    print("🟠 Naranja -> Voraz")
    print("🟣 Morado  -> Tabú")
    print("🩵 Cian    -> Recocido")
    print("--------------------------")

# -------------------------------
# -------------------------------
# -------------------------------
def main():
    origen = input("Origen: ")
    destino = input("Destino: ")


    directions = gmaps.directions(origen, destino, mode="driving", alternatives=True)

    nodos, conexiones = obtener_nodos(directions)
    G = construir_grafo(nodos, conexiones)

    inicio = 0
    fin = len(nodos) - 1

    print("Calculando...")


    ruta_bfs = bfs(G, inicio, fin, nodos)
    evitar = set(ruta_bfs) if ruta_bfs else set()


    ruta_dfs = dfs(G, inicio, fin, nodos, evitar)
    
    ruta_astar = astar(G, inicio, fin, nodos, evitar)
    ruta_voraz = voraz(G, inicio, fin, nodos)
    ruta_tabu = tabu(G, inicio, fin, nodos)
    ruta_recocido = recocido(G, inicio, fin, nodos)

    generar_mapa([
        ruta_bfs,
        ruta_dfs,
        ruta_astar,
        ruta_voraz,
        ruta_tabu,
        ruta_recocido
    ], nodos)

if __name__ == "__main__":
    main()