"""
SIMULACION GOOGLE MAPS - Proyecto Unidad 1
Solo ORIGEN y DESTINO. Nodos intermedios automaticos.
Multiples rutas alternativas. Cada algoritmo explora diferente.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_builder import GraphBuilder
from algorithms import SearchAlgorithms
from visualizer import RouteVisualizer


def banner():
    print("\n" + "=" * 60)
    print("  SIMULACION GOOGLE MAPS")
    print("  Proyecto Unidad 1 - IA")
    print("  OpenStreetMap + OSRM (gratuito)")
    print("  Cada algoritmo explora DIFERENTE")
    print("=" * 60)


def menu():
    print("\n" + "-" * 45)
    print("  MENU")
    print("-" * 45)
    print("   1. BFS  (anchura)")
    print("   2. DFS  (profundidad)")
    print("   3. ILDFS (profundidad iterativa)")
    print("   4. Voraz (greedy)")
    print("   5. A*   (optimo)")
    print("   6. Busqueda Tabu")
    print("   7. Recocido Simulado")
    print("   8. Comparar TODOS")
    print("   9. Ver mapa ultima ruta")
    print("  10. Cambiar ubicaciones")
    print("   0. Salir")
    print("-" * 45)


def pedir():
    print("\n  Ubicaciones:")
    print("   1. CDMX (Zocalo -> CU UNAM)")
    print("   2. Monterrey (Macroplaza -> BBVA)")
    print("   3. Madrid (Sol -> Bernabeu)")
    print("   4. Personalizado")

    op = input("\n  Opcion [1]: ").strip() or "1"

    ejemplos = {
        "1": ("Zocalo, Ciudad de Mexico, Mexico",
              "Ciudad Universitaria UNAM, Ciudad de Mexico"),
        "2": ("Macroplaza, Monterrey, Mexico",
              "Estadio BBVA, Guadalupe, Nuevo Leon, Mexico"),
        "3": ("Puerta del Sol, Madrid, Spain",
              "Santiago Bernabeu, Madrid, Spain"),
    }

    if op in ejemplos:
        o, d = ejemplos[op]
    else:
        o = input("  ORIGEN : ").strip()
        d = input("  DESTINO: ").strip()
        if not o or not d:
            o, d = ejemplos["1"]

    print(f"\n  ORIGEN : {o}")
    print(f"  DESTINO: {d}")
    return o, d


def construir(o, d):
    print("\n  Construyendo grafo (30-60 seg)...\n")
    gb = GraphBuilder()
    if not gb.build_graph(o, d):
        print("  ERROR. Verifica internet y direcciones.")
        return None
    return gb


def gen_mapa(viz, path, name, edges):
    fname = f"mapa_{name.replace(' ', '_')}.html"
    viz.create_map(
        path, name,
        explored_edges=edges,
        filename=fname
    )


def main():
    banner()
    o, d = pedir()
    gb = construir(o, d)
    if not gb:
        return

    algos = SearchAlgorithms(gb)
    viz = RouteVisualizer(gb)
    last_path = None
    last_name = ""
    last_edges = []

    while True:
        menu()
        op = input("  Opcion: ").strip()

        if op == "1":
            last_path = algos.bfs()
            last_name = "BFS"
            last_edges = list(algos.explored_edges)
        elif op == "2":
            last_path = algos.dfs()
            last_name = "DFS"
            last_edges = list(algos.explored_edges)
        elif op == "3":
            last_path = algos.ildfs()
            last_name = "ILDFS"
            last_edges = list(algos.explored_edges)
        elif op == "4":
            last_path = algos.greedy()
            last_name = "Voraz"
            last_edges = list(algos.explored_edges)
        elif op == "5":
            last_path = algos.a_star()
            last_name = "A_estrella"
            last_edges = list(algos.explored_edges)
        elif op == "6":
            last_path = algos.tabu_search()
            last_name = "Tabu"
            last_edges = list(algos.explored_edges)
        elif op == "7":
            last_path = algos.simulated_annealing()
            last_name = "Recocido"
            last_edges = list(algos.explored_edges)
        elif op == "8":
            res = algos.compare_all()
            valid = {k: v for k, v in res.items() if v['path']}
            if valid:
                w = min(valid,
                        key=lambda k: valid[k]['distance'])
                last_path = valid[w]['path']
                last_name = f"Mejor_{w}"
                last_edges = valid[w].get('edges', [])
        elif op == "9":
            if last_path:
                gen_mapa(viz, last_path, last_name, last_edges)
            else:
                print("\n  Ejecuta un algoritmo primero")
        elif op == "10":
            o, d = pedir()
            gb = construir(o, d)
            if not gb:
                continue
            algos = SearchAlgorithms(gb)
            viz = RouteVisualizer(gb)
            last_path = None
            last_name = ""
            last_edges = []
        elif op == "0":
            print("\n  Hasta luego!\n")
            break
        else:
            print("  Opcion invalida")
            continue

        if op in "1234567" and last_path:
            r = input("\n  Generar mapa? (s/n) [s]: "
                      ).strip().lower()
            if r != 'n':
                gen_mapa(viz, last_path, last_name, last_edges)


if __name__ == "__main__":
    main()