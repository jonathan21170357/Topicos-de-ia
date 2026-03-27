import googlemaps
import webbrowser
from collections import deque
import math
from heapq import heappush, heappop
import os
import polyline
import random
import json

API_KEY = "AIzaSyAUbNrDCnRiCqyIJ1lbe7l_8sryb2V3WUo"
gmaps = googlemaps.Client(key=API_KEY)

# -------------------------------
def distancia(a, b):
    # Heurística de Distancia Euclidiana
    return math.sqrt((a['lat']-b['lat'])**2 + (a['lng']-b['lng'])**2)

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
# ALGORITMOS DE BÚSQUEDA (Sin cambios)
# -------------------------------

def bfs(G, inicio, fin, nodos):
    cola = deque([[inicio]])
    visitados = set([inicio])
    while cola:
        ruta = cola.popleft()
        nodo = ruta[-1]
        if distancia(nodos[nodo], nodos[fin]) < 0.0005: return ruta
        for vecino in G[nodo]:
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(ruta + [vecino])
    return None

def dfs(G, inicio, fin, nodos, evitar):
    pila = [[inicio]]
    visitados = set()
    while pila:
        ruta = pila.pop()
        nodo = ruta[-1]
        if distancia(nodos[nodo], nodos[fin]) < 0.0005: return ruta
        if nodo not in visitados:
            visitados.add(nodo)
            vecinos = list(G[nodo])
            vecinos.sort(key=lambda x: 1 if x in evitar else 0, reverse=True)
            for vecino in vecinos:
                if vecino not in visitados: pila.append(ruta + [vecino])
    return None

def ldfs(G, inicio, fin, nodos, evitar, limite=2000):
    pila = [([inicio], 0)]  
    visitados_prof = {inicio: 0}
    while pila:
        ruta, prof = pila.pop()
        nodo = ruta[-1]
        if distancia(nodos[nodo], nodos[fin]) < 0.0005: return ruta
        if prof < limite:
            vecinos = list(G[nodo])
            vecinos.sort(key=lambda x: 1 if x in evitar else 0, reverse=True)
            for vecino in vecinos:
                if vecino not in visitados_prof or prof + 1 < visitados_prof[vecino]:
                    visitados_prof[vecino] = prof + 1
                    pila.append((ruta + [vecino], prof + 1))
    return None

def astar(G, inicio, fin, nodos, evitar):
    open_set = []
    heappush(open_set, (0, inicio))
    came_from = {}
    g_score = {inicio: 0}
    while open_set:
        _, actual = heappop(open_set)
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
                f = tentative_g + distancia(nodos[vecino], nodos[fin])
                heappush(open_set, (f, vecino))
    return None

def voraz(G, inicio, fin, nodos):
    actual = inicio
    ruta = [actual]
    visitados = set([inicio])
    while distancia(nodos[actual], nodos[fin]) >= 0.0005: 
        vecinos = [v for v in G[actual] if v not in visitados]
        if not vecinos:
            if len(ruta) > 1:
                ruta.pop(); actual = ruta[-1]; continue
            else: return None 
        siguiente = min(vecinos, key=lambda x: distancia(nodos[x], nodos[fin]))
        ruta.append(siguiente); visitados.add(siguiente); actual = siguiente
    return ruta

def tabu(G, inicio, fin, nodos):
    actual = inicio
    ruta = [actual]
    tabu_lista = set([inicio])
    while distancia(nodos[actual], nodos[fin]) >= 0.0005:
        vecinos = [v for v in G[actual] if v not in tabu_lista]
        if not vecinos:
            if len(ruta) > 1:
                ruta.pop(); actual = ruta[-1]; continue
            else: return None
        vecinos.sort(key=lambda x: distancia(nodos[x], nodos[fin]))
        siguiente = random.choice(vecinos[:3]) if len(vecinos) >= 3 else vecinos[0]
        ruta.append(siguiente); tabu_lista.add(siguiente); actual = siguiente
    return ruta

def recocido(G, inicio, fin, nodos):
    actual = inicio
    ruta = [actual]
    visitados = set([inicio])
    T = 1.0
    while distancia(nodos[actual], nodos[fin]) >= 0.0005:
        vecinos = [v for v in G[actual] if v not in visitados]
        if not vecinos:
            if len(ruta) > 1:
                ruta.pop(); actual = ruta[-1]; continue
            else: return None
        siguiente = random.choice(vecinos)
        delta = distancia(nodos[siguiente], nodos[fin]) - distancia(nodos[actual], nodos[fin])
        if delta < 0 or random.random() < math.exp(-delta / max(T, 0.001)):
            actual = siguiente; ruta.append(actual); visitados.add(actual)
        T *= 0.95 
    return ruta

# -------------------------------
# GENERAR MAPA
# -------------------------------
def generar_mapa(rutas_calculadas, nodos):
    

    datos_rutas_js = []
    nombres = ["BFS", "DFS", "A*", "Voraz", "Tabú", "Recocido", "LDFS"]
    colores = ["#0000FF", "#008000", "#FF0000", "#FFA500", "#800080", "#00FFFF", "#FF1493"]
    offsets = [0.00008, 0.00005, 0, -0.00005, -0.00008, 0.00011, -0.00011]

    ruta_valida_ejemplo = next((r for r in rutas_calculadas if r), None)
    if not ruta_valida_ejemplo:
        print("❌ No se encontró ninguna ruta válida para centrar el mapa.")
        return
    centro = nodos[ruta_valida_ejemplo[0]]

    # Serializar los datos de las rutas a JSON
    for i, ruta in enumerate(rutas_calculadas):
        if ruta:
            coordenadas = [{'lat': nodos[j]['lat'], 'lng': nodos[j]['lng']} for j in ruta]
            datos_rutas_js.append({
                'nombre': nombres[i],
                'color': colores[i],
                'offset': offsets[i],
                'path': coordenadas
            })

    json_datos = json.dumps(datos_rutas_js)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rutas Animadas Culiacán</title>
        <script src="https://maps.googleapis.com/maps/api/js?key={API_KEY}"></script>
        <style>
            #map {{ height: 100vh; width: 100%; }}
            #legend {{
                background: white; padding: 10px; margin: 10px;
                border: 1px solid #ccc; border-radius: 5px; font-family: Arial, sans-serif;
            }}
            .legend-item {{ margin-bottom: 5px; display: flex; align-items: center; }}
            .color-box {{ width: 20px; height: 10px; margin-right: 5px; border-radius: 2px; }}
        </style>
    </head>
    <body>
    <div id="map"></div>
    <div id="legend"><h4>Algoritmos</h4></div>

    <script>
    var map;
    var datosRutas = {json_datos}; // Importamos los datos calculados en Python

    function initMap() {{
        map = new google.maps.Map(document.getElementById('map'), {{
            zoom: 14,
            center: {{lat: {centro['lat']}, lng: {centro['lng']}}},
            mapTypeId: 'roadmap'
        }});

        // Añadir leyenda visual al mapa
        var legend = document.getElementById('legend');
        datosRutas.forEach(function(rutaData) {{
            var item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = '<div class="color-box" style="background:' + rutaData.color + ';"></div>' +
                             '<span>' + rutaData.nombre + '</span>';
            legend.appendChild(item);
        }});
        map.controls[google.maps.ControlPosition.RIGHT_BOTTOM].push(legend);

        // Marcador de Inicio Común
        if (datosRutas.length > 0 && datosRutas[0].path.length > 0) {{
            new google.maps.Marker({{
                position: datosRutas[0].path[0],
                map: map,
                label: 'S',
                title: 'Punto de Salida'
            }});
        }}

        // Iniciar la animación de cada ruta con un pequeño desfase entre algoritmos
        datosRutas.forEach(function(rutaData, index) {{
            // setTimeout para que no empiecen todos exactamente al mismo milisegundo visualmente
            setTimeout(function() {{
                animarRuta(rutaData);
            }}, index * 200); 
        }});
    }}

    function animarRuta(rutaData) {{
        var points = rutaData.path;
        var offset = rutaData.offset;
        var currentPath = [];
        
        // Crear la polilínea vacía
        var polyline = new google.maps.Polyline({{
            path: currentPath,
            geodesic: true,
            strokeColor: rutaData.color,
            strokeOpacity: 0.8,
            strokeWeight: 4,
            map: map
        }});

        var pointIndex = 0;
        
        // Función recursiva que usa setTimeout para dibujar
        function dibujarSiguientePunto() {{
            if (pointIndex >= points.length) {{
                // Añadir marcador de fin cuando termine la animación de esta ruta
                new google.maps.Marker({{
                    position: {{lat: points[points.length-1].lat + offset, lng: points[points.length-1].lng + offset}},
                    map: map,
                    icon: 'https://maps.google.com/mapfiles/kml/pal4/icon57.png', // Icono de bandera
                    title: 'Fin: ' + rutaData.nombre
                }});
                return; // Fin de la animación para esta ruta
            }}

            var nextPoint = points[pointIndex];
            // Aplicar el desfase visual (offset) directamente en JavaScript
            currentPath.push({{lat: nextPoint.lat + offset, lng: nextPoint.lng + offset}});
            polyline.setPath(currentPath);
            
            pointIndex++;

            // VELOCIDAD DE ANIMACIÓN: Controla el retraso entre puntos (en milisegundos)
            // 30ms es rápido, 100ms es lento. Ajusta aquí.
            setTimeout(dibujarSiguientePunto, 30); 
        }}

        dibujarSiguientePunto(); // Iniciar el bucle de dibujo
    }}

    window.onload = initMap;
    </script>
    </body>
    </html>
    """

    with open("mapa_animado.html", "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open("file://" + os.path.abspath("mapa_animado.html"))

    print("\n✅ PROCESO COMPLETADO")
    print("✔ Rutas calculadas e imprimieno distancias...")
    print("🌐 Abriendo 'mapa_animado.html' en tu navegador para ver la animación...")

# -------------------------------
# FUNCIONES DE MENÚ (Sin cambios)
# -------------------------------
def mostrar_menu():
    lugares = [
        "Parque las Riberas, Culiacán", "Catedral de Culiacán", "Plaza Sendero Culiacán",
        "Aeropuerto Internacional de Culiacán", "Instituto Tecnológico de Culiacán",
        "Ciudad Universitaria UAS, Culiacán", "Jardín Botánico Culiacán", "Templo de La Lomita, Culiacán"
    ]
    print("\n📍 --- LUGARES EN CULIACÁN --- 📍")
    for i, lugar in enumerate(lugares, 1): print(f"{i}. {lugar}")
    print("---------------------------------")
    return lugares

def obtener_seleccion(mensaje, max_opcion):
    while True:
        try:
            opcion = int(input(mensaje))
            if 1 <= opcion <= max_opcion: return opcion - 1
            else: print(f"⚠ Por favor, ingresa un número entre 1 y {max_opcion}.")
        except ValueError: print("⚠ Entrada no válida. Ingresa solo el número.")

# -------------------------------
# MAIN
# -------------------------------
def main():
    lugares = mostrar_menu()
    while True:
        idx_origen = obtener_seleccion("\nElige el número de tu ORIGEN: ", len(lugares))
        idx_destino = obtener_seleccion("Elige el número de tu DESTINO: ", len(lugares))
        if idx_origen == idx_destino: print("⚠ El origen y el destino no pueden ser el mismo.")
        else: break
            
    origen = lugares[idx_origen]; destino = lugares[idx_destino]
    print(f"\n🚗 Pidiendo rutas alternativas a Google Maps para '{origen}' -> '{destino}'...")
    directions = gmaps.directions(origen, destino, mode="driving", alternatives=True)
    
    if not directions:
        print("❌ Google Maps no encontró ruta."); return

    nodos, conexiones = obtener_nodos(directions)
    G = construir_grafo(nodos, conexiones)
    inicio = 0; fin = len(nodos) - 1

    print("Calculando caminos internamente en Python...")
    ruta_bfs = bfs(G, inicio, fin, nodos)
    evitar = set(ruta_bfs) if ruta_bfs else set()
    ruta_dfs = dfs(G, inicio, fin, nodos, evitar)
    ruta_ldfs = ldfs(G, inicio, fin, nodos, evitar, limite=2000)
    ruta_astar = astar(G, inicio, fin, nodos, evitar)
    ruta_voraz = voraz(G, inicio, fin, nodos)
    ruta_tabu = tabu(G, inicio, fin, nodos)
    ruta_recocido = recocido(G, inicio, fin, nodos)
    
    print("\n--- DISTANCIAS RECORRIDAS (Calculadas en Python) ---")
    nombres = ["BFS (Azul)", "DFS (Verde)", "A* (Rojo)", "Voraz (Naranja)", "Tabú (Morado)", "Recocido (Cian)", "LDFS (Rosa)"]
    rutas_calculadas = [ruta_bfs, ruta_dfs, ruta_astar, ruta_voraz, ruta_tabu, ruta_recocido, ruta_ldfs]

    for nombre, ruta in zip(nombres, rutas_calculadas):
        if ruta:
            dist_total = 0
            for i in range(len(ruta) - 1):
                dist_total += distancia(nodos[ruta[i]], nodos[ruta[i+1]]) * 111.1
            print(f"{nombre}: {dist_total:.2f} km")
        else: print(f"{nombre}: ❌ No encontró ruta")
    print("----------------------------------------------------\n")

    # Llamamos a la nueva función animada
    generar_mapa(rutas_calculadas, nodos)

if __name__ == "__main__":
    main()