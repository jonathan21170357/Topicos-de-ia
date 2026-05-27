"""
Construye un grafo con MULTIPLES rutas alternativas reales.
Cada ruta tiene DIFERENTE numero de nodos para que
BFS, DFS, Voraz, A* encuentren caminos distintos.
"""

import math
import requests
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


class GraphBuilder:
    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="proyecto_rutas_ia_final_v5", timeout=10
        )
        self.locations = {}
        self.addresses = {}
        self.graph = {}
        self.edges_info = {}
        self.all_polylines = {}
        self.direct_distance = 0
        self.direct_duration = 0
        self.direct_polyline = []
        self.route_polylines = []
        self.route_sequences = []

    def geocode_address(self, address):
        try:
            time.sleep(1.2)
            loc = self.geolocator.geocode(address)
            if loc:
                return (loc.latitude, loc.longitude)
        except GeocoderTimedOut:
            time.sleep(3)
            try:
                loc = self.geolocator.geocode(address)
                if loc:
                    return (loc.latitude, loc.longitude)
            except Exception:
                pass
        except Exception as e:
            print(f"      Error: {e}")
        return None

    def haversine(self, c1, c2):
        R = 6371
        la1, lo1 = math.radians(c1[0]), math.radians(c1[1])
        la2, lo2 = math.radians(c2[0]), math.radians(c2[1])
        a = (math.sin((la2 - la1) / 2) ** 2 +
             math.cos(la1) * math.cos(la2) *
             math.sin((lo2 - lo1) / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))

    def _create_fallback_routes(self, c1, c2):
        """Crea rutas artificiales cuando OSRM no responde"""
        print("     Usando rutas artificiales generadas localmente")
        routes = []
        direct_dist = self.haversine(c1, c2)
        
        # Ruta 1: línea recta
        routes.append({
            'distance': direct_dist,
            'duration': direct_dist * 2.5,
            'polyline': [c1, c2],
            'steps': []
        })
        
        # Ruta 2: con desvío hacia el norte
        lat_mid = (c1[0] + c2[0]) / 2
        lon_mid = (c1[1] + c2[1]) / 2
        offset = 0.03 * direct_dist / 10  # offset proporcional a la distancia
        
        north_point = (lat_mid + offset, lon_mid)
        routes.append({
            'distance': direct_dist * 1.25,
            'duration': direct_dist * 2.5 * 1.25,
            'polyline': [c1, north_point, c2],
            'steps': [{'coord': north_point, 'name': 'Desvío Norte'}]
        })
        
        # Ruta 3: con desvío hacia el sur
        south_point = (lat_mid - offset, lon_mid)
        routes.append({
            'distance': direct_dist * 1.3,
            'duration': direct_dist * 2.5 * 1.3,
            'polyline': [c1, south_point, c2],
            'steps': [{'coord': south_point, 'name': 'Desvío Sur'}]
        })
        
        # Ruta 4: ruta más larga con 2 desvíos
        way1 = (c1[0] + offset * 0.5, c1[1] - offset)
        way2 = (c2[0] - offset * 0.5, c2[1] + offset)
        routes.append({
            'distance': direct_dist * 1.5,
            'duration': direct_dist * 2.5 * 1.5,
            'polyline': [c1, way1, way2, c2],
            'steps': [
                {'coord': way1, 'name': 'Desvío Oeste'},
                {'coord': way2, 'name': 'Desvío Este'}
            ]
        })
        
        return routes

    def _osrm_alternatives(self, c1, c2):
        """Intenta obtener rutas de OSRM con reintentos"""
        # Probar con HTTPS primero
        urls = [
            f"https://router.project-osrm.org/route/v1/driving/{c1[1]},{c1[0]};{c2[1]},{c2[0]}?overview=full&geometries=geojson&steps=true&alternatives=true",
            f"http://router.project-osrm.org/route/v1/driving/{c1[1]},{c1[0]};{c2[1]},{c2[0]}?overview=full&geometries=geojson&steps=true&alternatives=true",
            f"https://routing.openstreetmap.de/route/v1/driving/{c1[1]},{c1[0]};{c2[1]},{c2[0]}?overview=full&geometries=geojson&steps=true&alternatives=true"
        ]
        
        for url_idx, url in enumerate(urls):
            for intento in range(2):  # 2 intentos por URL
                try:
                    if url_idx > 0 or intento > 0:
                        print(f"     Intentando alternativa {url_idx + 1}.{intento + 1}...")
                    data = requests.get(url, timeout=25)
                    if data.status_code == 200:
                        data_json = data.json()
                        if data_json.get('code') == 'Ok':
                            result = []
                            for r in data_json.get('routes', []):
                                poly = [(c[1], c[0])
                                        for c in r['geometry']['coordinates']]
                                steps = []
                                for leg in r.get('legs', []):
                                    for s in leg.get('steps', []):
                                        if s.get('maneuver'):
                                            loc = s['maneuver']['location']
                                            steps.append({
                                                'coord': (loc[1], loc[0]),
                                                'name': s.get('name', '')
                                            })
                                result.append({
                                    'distance': r['distance'] / 1000,
                                    'duration': r['duration'] / 60,
                                    'polyline': poly,
                                    'steps': steps
                                })
                            if result:
                                print(f"     Conectado exitosamente a OSRM")
                                return result
                except requests.exceptions.Timeout:
                    if intento == 1 and url_idx == len(urls) - 1:
                        print(f"     Timeout en todos los intentos")
                except Exception as e:
                    if intento == 1 and url_idx == len(urls) - 1:
                        print(f"     Error: {e}")
                time.sleep(1)
        
        print("     No se pudo conectar con OSRM, usando rutas artificiales")
        return self._create_fallback_routes(c1, c2)

    def _osrm_waypoint_route(self, origin, wp, dest):
        urls = [
            f"https://router.project-osrm.org/route/v1/driving/{origin[1]},{origin[0]};{wp[1]},{wp[0]};{dest[1]},{dest[0]}?overview=full&geometries=geojson&steps=true",
            f"http://router.project-osrm.org/route/v1/driving/{origin[1]},{origin[0]};{wp[1]},{wp[0]};{dest[1]},{dest[0]}?overview=full&geometries=geojson&steps=true"
        ]
        
        for url in urls:
            try:
                time.sleep(0.6)
                data = requests.get(url, timeout=15).json()
                if data.get('code') == 'Ok' and data.get('routes'):
                    r = data['routes'][0]
                    poly = [(c[1], c[0])
                            for c in r['geometry']['coordinates']]
                    steps = []
                    for leg in r.get('legs', []):
                        for s in leg.get('steps', []):
                            if s.get('maneuver'):
                                loc = s['maneuver']['location']
                                steps.append({
                                    'coord': (loc[1], loc[0]),
                                    'name': s.get('name', '')
                                })
                    return {
                        'distance': r['distance'] / 1000,
                        'duration': r['duration'] / 60,
                        'polyline': poly,
                        'steps': steps
                    }
            except Exception:
                pass
        return None

    def _extract_segment(self, polyline, c1, c2):
        if not polyline or len(polyline) < 2:
            return [c1, c2]
        i1 = min(range(len(polyline)),
                 key=lambda i: self.haversine(polyline[i], c1))
        i2 = min(range(len(polyline)),
                 key=lambda i: self.haversine(polyline[i], c2))
        lo, hi = min(i1, i2), max(i1, i2)
        seg = polyline[lo:hi + 1]
        return seg if len(seg) >= 2 else [c1, c2]

    def _add_edge(self, n1, n2, dist, polyline=None):
        dur = dist * 3
        self.graph.setdefault(n1, {})[n2] = dist
        self.graph.setdefault(n2, {})[n1] = dist
        info = {
            'distance_km': dist,
            'duration_min': dur,
            'distance_text': f"{dist:.1f} km",
            'duration_text': f"{int(dur)} min"
        }
        self.edges_info[(n1, n2)] = info
        self.edges_info[(n2, n1)] = info
        if polyline is None:
            polyline = [self.locations[n1], self.locations[n2]]
        self.all_polylines[(n1, n2)] = polyline
        self.all_polylines[(n2, n1)] = list(reversed(polyline))

    def build_graph(self, origin_addr, dest_addr):
        print("\n" + "=" * 60)
        print("  CONSTRUYENDO GRAFO CON RUTAS ALTERNATIVAS")
        print("  Fuente: OpenStreetMap + OSRM (gratuito)")
        print("=" * 60)

        # 1. Geocodificar
        print(f"\n  Geocodificando ORIGEN: {origin_addr}")
        oc = self.geocode_address(origin_addr)
        if not oc:
            print("     ERROR: no se pudo geocodificar")
            return False
        print(f"     OK ({oc[0]:.6f}, {oc[1]:.6f})")

        print(f"\n  Geocodificando DESTINO: {dest_addr}")
        dc = self.geocode_address(dest_addr)
        if not dc:
            print("     ERROR: no se pudo geocodificar")
            return False
        print(f"     OK ({dc[0]:.6f}, {dc[1]:.6f})")

        self.locations["ORIGEN"] = oc
        self.addresses["ORIGEN"] = origin_addr
        self.locations["DESTINO"] = dc
        self.addresses["DESTINO"] = dest_addr
        self.graph["ORIGEN"] = {}
        self.graph["DESTINO"] = {}

        # 2. Obtener rutas alternativas
        print(f"\n  Solicitando rutas a OSRM...")
        routes = self._osrm_alternatives(oc, dc)
        if not routes:
            print("     ERROR: no se pudieron obtener rutas")
            return False
            
        print(f"     Obtenidas {len(routes)} ruta(s)")

        # 3. Si faltan, generar alternativas adicionales
        if len(routes) < 3:
            print("     Generando alternativas adicionales...")
            dlat = dc[0] - oc[0]
            dlon = dc[1] - oc[1]
            mid = ((oc[0] + dc[0]) / 2, (oc[1] + dc[1]) / 2)
            for off in [0.3, -0.3, 0.15, -0.15, 0.4, -0.4]:
                if len(routes) >= 4:
                    break
                wp = (mid[0] + (-dlon) * off,
                      mid[1] + dlat * off)
                r = self._osrm_waypoint_route(oc, wp, dc)
                if r and r['distance'] < self.haversine(oc, dc) * 5:
                    dup = False
                    for ex in routes:
                        if abs(r['distance'] - ex['distance']) < 0.3:
                            dup = True
                            break
                    if not dup:
                        routes.append(r)
                        print(f"       + Alternativa: "
                              f"{r['distance']:.1f} km")

        self.direct_distance = routes[0]['distance']
        self.direct_duration = routes[0]['duration']
        self.direct_polyline = routes[0]['polyline']

        for r in routes:
            self.route_polylines.append(r['polyline'])

        print(f"\n  {len(routes)} rutas obtenidas:")
        for i, r in enumerate(routes):
            print(f"    Ruta {chr(65+i)}: {r['distance']:.1f} km, "
                  f"~{int(r['duration'])} min")

        # 4. Extraer nodos (DIFERENTE cantidad por ruta)
        print(f"\n  Extrayendo nodos intermedios...")

        # CLAVE: diferente numero de nodos por ruta
        # para que BFS (menos aristas) vs A* (menor distancia)
        # encuentren caminos distintos
        node_counts = [3, 5, 4, 6, 3, 5]

        for ri, route in enumerate(routes):
            letter = chr(65 + ri)
            count = node_counts[ri % len(node_counts)]

            valid = [s for s in route['steps']
                     if self.haversine(s['coord'], oc) > 0.25
                     and self.haversine(s['coord'], dc) > 0.25]

            if len(valid) > count:
                sp = len(valid) / count
                selected = [valid[int(i * sp)] for i in range(count)]
            else:
                selected = list(valid)

            # Si muy pocos, muestrear del polyline
            if len(selected) < 2:
                poly = route['polyline']
                for frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
                    idx = int(len(poly) * frac)
                    if idx >= len(poly):
                        idx = len(poly) - 1
                    coord = poly[idx]
                    if (self.haversine(coord, oc) > 0.25
                            and self.haversine(coord, dc) > 0.25):
                        selected.append({
                            'coord': coord,
                            'name': f'Punto ruta {letter}'
                        })
                    if len(selected) >= count:
                        break

            # Espaciado minimo
            filtered = []
            for s in selected:
                if (not filtered or
                        self.haversine(s['coord'],
                                       filtered[-1]['coord']) > 0.12):
                    filtered.append(s)

            # Crear nodos
            seq = ["ORIGEN"]
            for si, step in enumerate(filtered):
                name = f"R{letter}{si + 1}"
                self.locations[name] = step['coord']
                self.addresses[name] = (
                    f"[Ruta {letter}] {step['name']}"
                    if step.get('name')
                    else f"[Ruta {letter}] Nodo {si + 1}"
                )
                self.graph[name] = {}
                seq.append(name)
            seq.append("DESTINO")
            self.route_sequences.append(seq)
            print(f"    Ruta {letter} ({len(seq)-2} nodos): "
                  f"{' -> '.join(seq)}")

        # 5. Aristas secuenciales por ruta
        print(f"\n  Conectando nodos por ruta...")
        for ri, seq in enumerate(self.route_sequences):
            route = routes[ri]
            poly = route['polyline']

            total_h = sum(
                self.haversine(self.locations[seq[i]],
                               self.locations[seq[i + 1]])
                for i in range(len(seq) - 1)
            )
            scale = (route['distance'] / total_h
                     if total_h > 0.01 else 1.4)

            for i in range(len(seq) - 1):
                n1, n2 = seq[i], seq[i + 1]
                if n2 in self.graph.get(n1, {}):
                    continue
                h = self.haversine(self.locations[n1],
                                   self.locations[n2])
                road = h * scale
                segment = self._extract_segment(
                    poly, self.locations[n1], self.locations[n2]
                )
                self._add_edge(n1, n2, road, segment)
                print(f"     {n1} -> {n2}: {road:.2f} km")

        # 6. Conexiones cruzadas SOLO si estan muy cerca
        print(f"\n  Buscando cruces entre rutas...")
        membership = {}
        for ri, seq in enumerate(self.route_sequences):
            for n in seq:
                membership.setdefault(n, set()).add(ri)

        intermedios = [n for n in self.locations
                       if n not in ("ORIGEN", "DESTINO")]
        cruces = 0

        for i in range(len(intermedios)):
            n1 = intermedios[i]
            for j in range(i + 1, len(intermedios)):
                n2 = intermedios[j]
                if membership.get(n1) == membership.get(n2):
                    continue
                d = self.haversine(self.locations[n1],
                                   self.locations[n2])
                if d < 0.6 and n2 not in self.graph.get(n1, {}):
                    self._add_edge(n1, n2, d * 1.4)
                    cruces += 1
                    print(f"     {n1} <-> {n2}: "
                          f"{d * 1.4:.2f} km (cruce)")

        print(f"     {cruces} cruces entre rutas")

        # 7. Conectividad
        self._ensure_connectivity()

        self._print_summary()
        return True

    def _ensure_connectivity(self):
        visited = set()
        queue = ["ORIGEN"]
        visited.add("ORIGEN")
        while queue:
            node = queue.pop(0)
            for nb in self.graph.get(node, {}):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        unreachable = set(self.locations.keys()) - visited
        if unreachable:
            print(f"\n  Conectando {len(unreachable)} nodo(s) aislados")
            for node in unreachable:
                closest = min(
                    visited,
                    key=lambda n: self.haversine(
                        self.locations[node], self.locations[n])
                )
                d = self.haversine(self.locations[node],
                                   self.locations[closest]) * 1.4
                self._add_edge(node, closest, d)
                visited.add(node)

    def _print_summary(self):
        edges = sum(len(v) for v in self.graph.values()) // 2
        print(f"\n" + "=" * 60)
        print(f"  GRAFO: {len(self.locations)} nodos, {edges} aristas")
        print(f"  Directa: {self.direct_distance:.1f} km")
        print(f"  Rutas: {len(self.route_sequences)}")
        print("=" * 60)
        for seq in self.route_sequences:
            print(f"    {' -> '.join(seq)}")
        print()
        for n in sorted(self.graph.keys()):
            nbs = len(self.graph[n])
            print(f"  [{n}] {self.addresses.get(n, '')} "
                  f"({nbs} vecinos)")
            for nb, d in sorted(self.graph[n].items(),
                                key=lambda x: x[1]):
                print(f"      -> {nb}: {d:.2f} km")

    def get_heuristic(self, node, goal="DESTINO"):
        if node not in self.locations or goal not in self.locations:
            return float('inf')
        return self.haversine(self.locations[node],
                              self.locations[goal])