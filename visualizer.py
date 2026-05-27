"""
Mapa interactivo que muestra:
- Rutas alternativas de fondo (colores, punteadas)
- Aristas del grafo (gris)
- Exploracion del algoritmo (NARANJA)
- Camino final encontrado (AZUL grueso)
- Nodos con marcadores
"""

import folium
import os
import webbrowser


class RouteVisualizer:
    def __init__(self, graph_builder):
        self.gb = graph_builder

    def create_map(self, path=None, algorithm_name="",
                   explored_edges=None,
                   show_all_edges=True,
                   filename="ruta_mapa.html",
                   open_browser=True):

        if not self.gb.locations:
            print("  Sin ubicaciones")
            return None

        coords = list(self.gb.locations.values())
        center = [sum(c[0] for c in coords) / len(coords),
                  sum(c[1] for c in coords) / len(coords)]

        m = folium.Map(location=center, zoom_start=13,
                       tiles='OpenStreetMap')

        # 1. Rutas alternativas de fondo
        alt_colors = ['#4CAF50', '#FF9800', '#9C27B0',
                      '#00BCD4', '#795548']
        for i, poly in enumerate(self.gb.route_polylines):
            c = alt_colors[i % len(alt_colors)]
            folium.PolyLine(
                poly, weight=3, color=c, opacity=0.25,
                dash_array='8 6',
                tooltip=f"Ruta alternativa {chr(65 + i)}"
            ).add_to(m)

        # 2. Aristas del grafo
        if show_all_edges:
            drawn = set()
            for (n1, n2), poly in self.gb.all_polylines.items():
                key = tuple(sorted([n1, n2]))
                if key in drawn:
                    continue
                drawn.add(key)
                info = self.gb.edges_info.get((n1, n2), {})
                folium.PolyLine(
                    poly, weight=1.5, color='#9E9E9E',
                    opacity=0.3, dash_array='3 3',
                    tooltip=(f"{n1}<->{n2}: "
                             f"{info.get('distance_text', '?')}")
                ).add_to(m)

        # 3. EXPLORACION del algoritmo (NARANJA)
        if explored_edges:
            drawn_exp = set()
            for a, b in explored_edges:
                key = tuple(sorted([a, b]))
                if key in drawn_exp:
                    continue
                drawn_exp.add(key)
                poly = self.gb.all_polylines.get(
                    (a, b),
                    self.gb.all_polylines.get(
                        (b, a),
                        [self.gb.locations.get(a, (0, 0)),
                         self.gb.locations.get(b, (0, 0))]
                    )
                )
                folium.PolyLine(
                    poly, weight=4, color='#FF6F00',
                    opacity=0.6,
                    tooltip=f"Explorado: {a} -> {b}"
                ).add_to(m)

        # 4. CAMINO FINAL (AZUL grueso)
        path_colors = ['#1565C0', '#0D47A1', '#4A148C',
                       '#B71C1C', '#E65100', '#1B5E20',
                       '#283593']
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                n1, n2 = path[i], path[i + 1]
                poly = self.gb.all_polylines.get(
                    (n1, n2),
                    [self.gb.locations[n1],
                     self.gb.locations[n2]]
                )
                color = path_colors[i % len(path_colors)]
                info = self.gb.edges_info.get((n1, n2), {})

                folium.PolyLine(
                    poly, weight=7, color=color, opacity=0.9,
                    tooltip=(f"Paso {i + 1}: {n1}->{n2} "
                             f"({info.get('distance_text', '?')})")
                ).add_to(m)

                mid = poly[len(poly) // 2]
                folium.Marker(
                    mid,
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="background:{color};'
                            f'color:white;border-radius:50%;'
                            f'width:24px;height:24px;'
                            f'text-align:center;'
                            f'line-height:24px;'
                            f'font-size:12px;font-weight:bold;'
                            f'box-shadow:0 2px 4px '
                            f'rgba(0,0,0,.4);">'
                            f'{i + 1}</div>'
                        ),
                        icon_size=(24, 24),
                        icon_anchor=(12, 12)
                    )
                ).add_to(m)

        # 5. Marcadores de nodos
        for name, coord in self.gb.locations.items():
            if name == "ORIGEN":
                color, icon = 'green', 'home'
            elif name == "DESTINO":
                color, icon = 'red', 'flag'
            else:
                color, icon = 'blue', 'map-marker'

            addr = self.gb.addresses.get(name, '')
            folium.Marker(
                coord,
                popup=(f"<b>{name}</b><br>{addr}<br>"
                       f"<small>({coord[0]:.5f}, "
                       f"{coord[1]:.5f})</small>"),
                tooltip=f"{name}",
                icon=folium.Icon(color=color, icon=icon,
                                 prefix='fa')
            ).add_to(m)

        # Titulo
        if path:
            from algorithms import SearchAlgorithms
            sa = SearchAlgorithms(self.gb)
            d = sa.path_dist(path)
            title = (f"{algorithm_name} | {d:.2f} km | "
                     f"{' -> '.join(path)}")
        else:
            title = "Grafo de navegacion"

        m.get_root().html.add_child(folium.Element(f'''
        <div style="position:fixed;top:10px;left:50%;
            transform:translateX(-50%);z-index:1000;
            background:white;padding:8px 16px;
            border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,.3);
            font-family:Arial;font-size:13px;
            max-width:92%;">
            <b>{title}</b>
        </div>'''))

        m.get_root().html.add_child(folium.Element('''
        <div style="position:fixed;bottom:30px;left:20px;
            z-index:1000;background:white;padding:10px 14px;
            border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,.3);
            font-family:Arial;font-size:11px;
            line-height:2;">
            <b>Leyenda</b><br>
            <span style="color:green;">&#9679;</span> Origen /
            <span style="color:red;">&#9679;</span> Destino /
            <span style="color:blue;">&#9679;</span> Nodo<br>
            <span style="color:#1565C0;">&#9473;&#9473;&#9473;</span>
              Camino encontrado<br>
            <span style="color:#FF6F00;">&#9473;&#9473;&#9473;</span>
              Exploracion del algoritmo<br>
            <span style="color:#4CAF50;">- - -</span>
            <span style="color:#FF9800;">- - -</span>
            <span style="color:#9C27B0;">- - -</span>
              Rutas alternativas<br>
            <span style="color:#9E9E9E;">- - -</span>
              Aristas del grafo
        </div>'''))

        m.save(filename)
        abs_path = os.path.abspath(filename)
        print(f"\n  Mapa: {abs_path}")

        if open_browser:
            try:
                webbrowser.open('file://' + abs_path)
                print("  Abriendo navegador...")
            except Exception:
                print("  Abre el HTML manualmente")

        return m