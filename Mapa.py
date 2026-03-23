import googlemaps
import webbrowser
from collections import deque
import math
from heapq import heappush, heappop
import os
import polyline

API_KEY = "AIzaSyAUbNrDCnRiCqyIJ1lbe7l_8sryb2V3WUo"
gmaps = googlemaps.Client(key=API_KEY)

# -------------------------------
# Distancia (heurística)
# -------------------------------
def distancia(a, b):
    return math.sqrt((a['lat']-b['lat'])**2 + (a['lng']-b['lng'])**2)

# -------------------------------
# Construir grafo con bifurcaciones
# -------------------------------
def construir_grafo(steps):
    G = {}
    for i in range(len(steps)):
        G.setdefault(i, [])

        if i+1 < len(steps):
            G[i].append(i+1)

        if i+2 < len(steps):
            G[i].append(i+2)

        if i+3 < len(steps):
            G[i].append(i+3)

    return G

# -------------------------------
# Convertir a ruta real (calles)
# -------------------------------
def obtener_ruta_real(steps, indices_ruta):
    ruta_real = []

    for i in indices_ruta:
        puntos = polyline.decode(steps[i]['polyline']['points'])
        for lat, lng in puntos:
            ruta_real.append({'lat': lat, 'lng': lng})

    return ruta_real

# -------------------------------
# BFS
# -------------------------------
def bfs(G, inicio, fin):
    cola = deque([[inicio]])
    visitados = set([inicio])
    pasos = []

    while cola:
        ruta = cola.popleft()
        nodo = ruta[-1]

        pasos.append(ruta)

        if nodo == fin:
            return ruta, pasos

        for vecino in G.get(nodo, []):
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(ruta + [vecino])

    return None, pasos

# -------------------------------
# DFS
# -------------------------------
def dfs(G, inicio, fin):
    pila = [[inicio]]
    visitados = set([inicio])
    pasos = []

    while pila:
        ruta = pila.pop()
        nodo = ruta[-1]

        pasos.append(ruta)

        if nodo == fin:
            return ruta, pasos

        for vecino in G.get(nodo, []):
            if vecino not in visitados:
                visitados.add(vecino)
                pila.append(ruta + [vecino])

    return None, pasos

# -------------------------------
# A*
# -------------------------------
def astar(G, inicio, fin, steps):
    open_set = []
    heappush(open_set, (0, inicio))

    came_from = {}
    g_score = {inicio: 0}
    pasos = []

    while open_set:
        _, actual = heappop(open_set)

        # reconstruir ruta parcial
        ruta_temp = []
        n = actual
        while n in came_from:
            ruta_temp.append(n)
            n = came_from[n]
        ruta_temp.append(inicio)
        ruta_temp.reverse()

        pasos.append(ruta_temp)

        if actual == fin:
            return ruta_temp, pasos

        for vecino in G.get(actual, []):
            tentative_g = g_score[actual] + 1

            if vecino not in g_score or tentative_g < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = tentative_g

                h = distancia(
                    steps[vecino]['start_location'],
                    steps[fin]['end_location']
                )

                f = tentative_g + h
                heappush(open_set, (f, vecino))

    return None, pasos

# -------------------------------
# Generar mapa
# -------------------------------
def generar_mapa(ruta_indices, pasos_indices, steps, origen, destino, algoritmo):
    ruta_real = obtener_ruta_real(steps, ruta_indices)

    pasos_reales = [obtener_ruta_real(steps, p) for p in pasos_indices]

    centro = ruta_real[0]

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ruta</title>
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

        var pasos = [];
    """

    # pasos exploración
    for ruta_parcial in pasos_reales:
        path = [f"{{lat:{p['lat']}, lng:{p['lng']}}}" for p in ruta_parcial]
        html += f"pasos.push([{','.join(path)}]);\n"

    html += """
        var i = 0;

        function dibujarPaso() {
            if (i < pasos.length) {
                new google.maps.Polyline({
                    path: pasos[i],
                    strokeColor: "#AAAAAA",
                    strokeOpacity: 0.5,
                    strokeWeight: 2,
                    map: map
                });
                i++;
                setTimeout(dibujarPaso, 300);
            } else {
                new google.maps.Polyline({
                    path: pasos[pasos.length - 1],
                    strokeColor: "#FF0000",
                    strokeOpacity: 1.0,
                    strokeWeight: 4,
                    map: map
                });
            }
        }

        dibujarPaso();
    }

    window.onload = initMap;
    </script>
    </body>
    </html>
    """

    with open("ruta.html", "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open("file://" + os.path.abspath("ruta.html"))

    print("\n" + "="*40)
    print("✅ MAPA GENERADO CORRECTAMENTE")
    print("="*40)
    print(f"Punto A: {origen}")
    print(f"Punto B: {destino}")
    print(f"Algoritmo utilizado: {algoritmo}")
    print("="*40)

# -------------------------------
# MAIN
# -------------------------------
def main():
    origen = input("Dirección origen: ")
    destino = input("Dirección destino: ")

    directions = gmaps.directions(origen, destino, mode="driving")

    if not directions:
        print("No se pudo obtener la ruta")
        return

    steps = directions[0]['legs'][0]['steps']

    G = construir_grafo(steps)
    inicio, fin = 0, len(steps) - 1

    print("\nALGORITMOS:")
    print("1 BFS")
    print("2 DFS")
    print("3 A*")

    try:
        op = int(input("Selecciona algoritmo: ").strip())
    except:
        print("Opción inválida")
        return

    if op == 1:
        ruta, pasos = bfs(G, inicio, fin)
        algoritmo = "BFS"
    elif op == 2:
        ruta, pasos = dfs(G, inicio, fin)
        algoritmo = "DFS"
    elif op == 3:
        ruta, pasos = astar(G, inicio, fin, steps)
        algoritmo = "A*"
    else:
        print("Opción inválida")
        return

    if ruta is None:
        print("No se encontró ruta")
    else:
        print("Generando mapa...")
        generar_mapa(ruta, pasos, steps, origen, destino, algoritmo)

# -------------------------------
if __name__ == "__main__":
    main()