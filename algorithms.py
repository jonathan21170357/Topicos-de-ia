"""
7 algoritmos de busqueda.
CADA UNO EXPLORA DE FORMA DIFERENTE:
- BFS/DFS: NO ordenan vecinos por distancia
- BFS usa FIFO, DFS usa LIFO -> exploran rutas distintas
- Voraz/A* usan heuristica/costo
- Tabu/SA son estocasticos
Se registra la exploracion para mostrarla en el mapa.
"""

from collections import deque
import heapq
import random
import math
import time


class SearchAlgorithms:
    def __init__(self, graph_builder):
        self.gb = graph_builder
        self.graph = graph_builder.graph
        self.explored_edges = []
        self.explored_nodes = []

    def reset_trace(self):
        self.explored_edges = []
        self.explored_nodes = []

    def _step(self, num, msg, **kw):
        print(f"\n  {'─' * 50}")
        print(f"  Paso {num}: {msg}")
        if kw.get('node'):
            a = self.gb.addresses.get(kw['node'], '')
            print(f"    Nodo    : {kw['node']} ({a})")
        if kw.get('path'):
            print(f"    Camino  : {' -> '.join(kw['path'])}")
        if kw.get('frontier') is not None:
            f = kw['frontier']
            txt = str(f[:8]) + (f"...+{len(f)-8}" if len(f) > 8 else "")
            print(f"    Frontera: {txt}")
        if kw.get('visited') is not None:
            print(f"    Visitados: {sorted(kw['visited'])}")
        if kw.get('info'):
            for k, v in kw['info'].items():
                print(f"    {k}: {v}")

    def path_dist(self, path):
        t = 0
        for i in range(len(path) - 1):
            d = self.graph.get(path[i], {}).get(path[i + 1])
            if d is None:
                return float('inf')
            t += d
        return t

    def _result(self, name, path, explored):
        print(f"\n  {'=' * 50}")
        if path:
            d = self.path_dist(path)
            print(f"  RESULTADO - {name}")
            print(f"  Camino    : {' -> '.join(path)}")
            print(f"  Distancia : {d:.2f} km")
            print(f"  Nodos exp.: {explored}")
            print(f"  Saltos    : {len(path) - 1}")
            print(f"  Directa   : {self.gb.direct_distance:.2f} km")
            diff = d - self.gb.direct_distance
            print(f"  Diferencia: {diff:+.2f} km")
            print(f"\n  Tramos:")
            for i in range(len(path) - 1):
                dd = self.graph[path[i]][path[i + 1]]
                print(f"    {i+1}. {path[i]} -> {path[i+1]}: "
                      f"{dd:.2f} km")
        else:
            print(f"  {name}: SIN CAMINO")
        print(f"  {'=' * 50}")
        return path

    # ═══════════════════════════════════════════
    # 1. BFS - Busqueda en Anchura
    # ═══════════════════════════════════════════
    def bfs(self, start="ORIGEN", goal="DESTINO"):
        self.reset_trace()
        print("\n" + "=" * 60)
        print("  1. BFS - BUSQUEDA EN ANCHURA")
        print("=" * 60)
        print("  Cola FIFO. Nivel por nivel.")
        print("  Encuentra camino con MENOS ARISTAS.")
        print("  Vecinos en orden de insercion (NO por distancia).")

        queue = deque([(start, [start])])
        visited = {start}
        step = 0
        explored = 0

        while queue:
            cur, path = queue.popleft()
            explored += 1
            step += 1
            self.explored_nodes.append(cur)

            self._step(step, f"Desencolando '{cur}'",
                       node=cur, path=path,
                       frontier=[p[0] for p in list(queue)[:8]],
                       visited=visited)

            if cur == goal:
                print(f"\n  DESTINO ENCONTRADO!")
                return self._result("BFS", path, explored)

            # SIN ORDENAR - orden natural de insercion
            for vec, dist in self.graph[cur].items():
                if vec not in visited:
                    visited.add(vec)
                    queue.append((vec, path + [vec]))
                    self.explored_edges.append((cur, vec))
                    print(f"      + Encolando '{vec}' ({dist:.2f} km)")
                else:
                    print(f"      - '{vec}' ya visitado")

        return self._result("BFS", None, explored)

    # ═══════════════════════════════════════════
    # 2. DFS - Busqueda en Profundidad
    # ═══════════════════════════════════════════
    def dfs(self, start="ORIGEN", goal="DESTINO"):
        self.reset_trace()
        print("\n" + "=" * 60)
        print("  2. DFS - BUSQUEDA EN PROFUNDIDAD")
        print("=" * 60)
        print("  Pila LIFO. Va lo MAS PROFUNDO posible.")
        print("  Vecinos en orden natural -> LIFO invierte.")
        print("  Explora la ULTIMA ruta primero (diferente a BFS).")

        stack = [(start, [start])]
        visited = set()
        step = 0
        explored = 0

        while stack:
            cur, path = stack.pop()
            step += 1

            if cur in visited:
                continue

            visited.add(cur)
            explored += 1
            self.explored_nodes.append(cur)

            self._step(step, f"Desapilando '{cur}'",
                       node=cur, path=path,
                       frontier=[p[0] for p in stack[-5:]],
                       visited=visited)

            if cur == goal:
                print(f"\n  DESTINO ENCONTRADO!")
                return self._result("DFS", path, explored)

            # Orden natural de insercion (LIFO toma el ultimo)
            for vec, dist in self.graph[cur].items():
                if vec not in visited:
                    stack.append((vec, path + [vec]))
                    self.explored_edges.append((cur, vec))
                    print(f"      + Apilando '{vec}' ({dist:.2f} km)")

        return self._result("DFS", None, explored)

    # ═══════════════════════════════════════════
    # 3. ILDFS - Profundidad Iterativa
    # ═══════════════════════════════════════════
    def ildfs(self, start="ORIGEN", goal="DESTINO"):
        self.reset_trace()
        print("\n" + "=" * 60)
        print("  3. ILDFS - PROFUNDIDAD ITERATIVA")
        print("=" * 60)
        print("  DFS con limite creciente de profundidad.")

        total = 0
        for limit in range(1, len(self.graph) + 1):
            print(f"\n  {'━' * 45}")
            print(f"  Limite = {limit}")
            r, exp = self._ldfs(start, goal, limit)
            total += exp
            if r:
                return self._result("ILDFS", r, total)
            print(f"     Sin solucion con prof. {limit}")

        return self._result("ILDFS", None, total)

    def _ldfs(self, start, goal, limit):
        stack = [(start, [start], 0)]
        visited = set()
        explored = 0
        step = 0

        while stack:
            cur, path, depth = stack.pop()
            step += 1
            if cur in visited:
                continue
            visited.add(cur)
            explored += 1
            self.explored_nodes.append(cur)

            self._step(step, f"'{cur}' prof {depth}/{limit}",
                       node=cur, path=path, visited=visited)

            if cur == goal:
                return path, explored

            if depth < limit:
                for vec, dist in self.graph[cur].items():
                    if vec not in visited:
                        stack.append((vec, path + [vec], depth + 1))
                        self.explored_edges.append((cur, vec))
                        print(f"        + '{vec}' (p.{depth + 1})")
            else:
                print(f"        Limite alcanzado")

        return None, explored

    # ═══════════════════════════════════════════
    # 4. VORAZ - Greedy Best-First
    # ═══════════════════════════════════════════
    def greedy(self, start="ORIGEN", goal="DESTINO"):
        self.reset_trace()
        print("\n" + "=" * 60)
        print("  4. VORAZ - GREEDY BEST-FIRST")
        print("=" * 60)
        print("  Siempre expande el nodo con MENOR h(n).")
        print("  h = distancia en linea recta al destino.")

        cnt = 0
        h0 = self.gb.get_heuristic(start, goal)
        heap = [(h0, cnt, start, [start])]
        visited = set()
        step = 0
        explored = 0

        while heap:
            h, _, cur, path = heapq.heappop(heap)
            step += 1
            if cur in visited:
                continue
            visited.add(cur)
            explored += 1
            self.explored_nodes.append(cur)

            self._step(step, f"Expandiendo '{cur}'",
                       node=cur, path=path, visited=visited,
                       info={"h(n)": f"{h:.2f} km"})

            if cur == goal:
                return self._result("Voraz", path, explored)

            for vec, dist in self.graph[cur].items():
                if vec not in visited:
                    hn = self.gb.get_heuristic(vec, goal)
                    cnt += 1
                    heapq.heappush(heap,
                                   (hn, cnt, vec, path + [vec]))
                    self.explored_edges.append((cur, vec))
                    print(f"      + '{vec}': h={hn:.2f} km")

        return self._result("Voraz", None, explored)

    # ═══════════════════════════════════════════
    # 5. A* - A Estrella
    # ═══════════════════════════════════════════
    def a_star(self, start="ORIGEN", goal="DESTINO"):
        self.reset_trace()
        print("\n" + "=" * 60)
        print("  5. A* - A ESTRELLA")
        print("=" * 60)
        print("  f(n) = g(n) + h(n)")
        print("  Encuentra el camino con MENOR DISTANCIA REAL.")

        cnt = 0
        h0 = self.gb.get_heuristic(start, goal)
        heap = [(h0, cnt, start, [start], 0)]
        g_best = {start: 0}
        visited = set()
        step = 0
        explored = 0

        while heap:
            f, _, cur, path, g = heapq.heappop(heap)
            step += 1
            if cur in visited:
                continue
            visited.add(cur)
            explored += 1
            self.explored_nodes.append(cur)

            hc = self.gb.get_heuristic(cur, goal)
            self._step(step, f"Expandiendo '{cur}'",
                       node=cur, path=path, visited=visited,
                       info={"g": f"{g:.2f}", "h": f"{hc:.2f}",
                             "f": f"{f:.2f} km"})

            if cur == goal:
                print(f"\n  CAMINO OPTIMO!")
                return self._result("A*", path, explored)

            for vec, dist in self.graph[cur].items():
                if vec not in visited:
                    ng = g + dist
                    if vec not in g_best or ng < g_best[vec]:
                        g_best[vec] = ng
                        hn = self.gb.get_heuristic(vec, goal)
                        fn = ng + hn
                        cnt += 1
                        heapq.heappush(
                            heap, (fn, cnt, vec, path + [vec], ng)
                        )
                        self.explored_edges.append((cur, vec))
                        print(f"      + '{vec}': g={ng:.2f} "
                              f"h={hn:.2f} f={fn:.2f}")

        return self._result("A*", None, explored)

    # ═══════════════════════════════════════════
    # 6. BUSQUEDA TABU
    # ═══════════════════════════════════════════
    def tabu_search(self, start="ORIGEN", goal="DESTINO",
                    max_iter=120, tabu_size=20):
        self.reset_trace()
        print("\n" + "=" * 60)
        print("  6. BUSQUEDA TABU")
        print("=" * 60)
        print(f"  Busqueda local + lista tabu.")
        print(f"  iter={max_iter}, tabu={tabu_size}")

        cur_path = self._greedy_init(start, goal)
        if not cur_path:
            return self._result("Tabu", None, 0)

        cc = self.path_dist(cur_path)
        best = cur_path[:]
        bc = cc
        tabu = deque(maxlen=tabu_size)
        explored = 0

        self.explored_nodes = list(cur_path)
        for i in range(len(cur_path) - 1):
            self.explored_edges.append(
                (cur_path[i], cur_path[i + 1]))

        print(f"\n  Inicial: {' -> '.join(cur_path)} ({cc:.2f} km)")

        for it in range(1, max_iter + 1):
            explored += 1
            nbs = self._gen_neighbors(cur_path, start, goal)
            if not nbs:
                break

            best_nb = None
            bnc = float('inf')
            for nb in nbs:
                c = self.path_dist(nb)
                if c < bc:  # aspiracion
                    best_nb, bnc = nb, c
                    break
                if tuple(nb) not in tabu and c < bnc:
                    best_nb, bnc = nb, c

            if not best_nb:
                break

            tabu.append(tuple(cur_path))
            cur_path = best_nb
            cc = bnc

            for n in cur_path:
                if n not in self.explored_nodes:
                    self.explored_nodes.append(n)
            for i in range(len(cur_path) - 1):
                e = (cur_path[i], cur_path[i + 1])
                if e not in self.explored_edges:
                    self.explored_edges.append(e)

            imp = ""
            if cc < bc:
                best = cur_path[:]
                bc = cc
                imp = " <<< MEJORA >>>"

            if it <= 10 or it % 15 == 0 or imp:
                print(f"  It {it:>3}: {' -> '.join(cur_path)} "
                      f"({cc:.2f} km){imp}")

        return self._result("Busqueda Tabu", best, explored)

    # ═══════════════════════════════════════════
    # 7. RECOCIDO SIMULADO
    # ═══════════════════════════════════════════
    def simulated_annealing(self, start="ORIGEN", goal="DESTINO",
                            t0=500, alpha=0.92, tmin=0.1,
                            maxiter=300):
        self.reset_trace()
        print("\n" + "=" * 60)
        print("  7. RECOCIDO SIMULADO")
        print("=" * 60)
        print(f"  T0={t0}, alpha={alpha}, Tmin={tmin}")

        cur_path = self._greedy_init(start, goal)
        if not cur_path:
            return self._result("Recocido Simulado", None, 0)

        cc = self.path_dist(cur_path)
        best = cur_path[:]
        bc = cc
        temp = t0
        explored = 0

        self.explored_nodes = list(cur_path)
        for i in range(len(cur_path) - 1):
            self.explored_edges.append(
                (cur_path[i], cur_path[i + 1]))

        print(f"\n  Inicial: {' -> '.join(cur_path)} ({cc:.2f} km)")

        for it in range(1, maxiter + 1):
            if temp < tmin:
                break
            explored += 1

            nb = self._rand_neighbor(cur_path, start, goal)
            if not nb:
                temp *= alpha
                continue

            nc = self.path_dist(nb)
            delta = nc - cc

            if delta < 0:
                accept = True
            else:
                p = math.exp(-delta / temp) if temp > 0 else 0
                accept = random.random() < p

            if accept:
                cur_path = nb
                cc = nc

                for n in cur_path:
                    if n not in self.explored_nodes:
                        self.explored_nodes.append(n)
                for i in range(len(cur_path) - 1):
                    e = (cur_path[i], cur_path[i + 1])
                    if e not in self.explored_edges:
                        self.explored_edges.append(e)

                imp = ""
                if cc < bc:
                    best = cur_path[:]
                    bc = cc
                    imp = " <<< MEJOR >>>"

                if it <= 5 or it % 30 == 0 or imp:
                    print(f"  It {it:>3} T={temp:>7.1f} "
                          f"d={delta:>+.2f} "
                          f"{' -> '.join(cur_path)} "
                          f"({cc:.2f}){imp}")

            temp *= alpha

        return self._result("Recocido Simulado", best, explored)

    # ───────────────────────────────────────────
    # Auxiliares
    # ───────────────────────────────────────────
    def _greedy_init(self, s, g):
        path = [s]
        cur = s
        vis = {s}
        while cur != g:
            cands = [(n, d) for n, d in self.graph[cur].items()
                     if n not in vis]
            if not cands:
                if g in self.graph.get(cur, {}):
                    path.append(g)
                    return path
                return None
            best = min(cands,
                       key=lambda x: self.gb.get_heuristic(x[0], g))
            path.append(best[0])
            vis.add(best[0])
            cur = best[0]
        return path

    def _valid(self, p):
        for i in range(len(p) - 1):
            if p[i + 1] not in self.graph.get(p[i], {}):
                return False
        return True

    def _gen_neighbors(self, path, s, g):
        nbs = []
        mid = path[1:-1]
        # Intercambios
        for i in range(len(mid)):
            for j in range(i + 1, len(mid)):
                new = path[:]
                new[i + 1], new[j + 1] = new[j + 1], new[i + 1]
                if self._valid(new):
                    nbs.append(new)
        # Eliminaciones
        for i in range(len(mid)):
            new = path[:i + 1] + path[i + 2:]
            if self._valid(new):
                nbs.append(new)
        # Inserciones
        unused = set(self.graph.keys()) - set(path)
        for node in unused:
            for i in range(1, len(path)):
                new = path[:i] + [node] + path[i:]
                if self._valid(new):
                    nbs.append(new)
        # Intercambio con insercion
        for i in range(len(mid)):
            for node in set(self.graph.keys()) - set(path):
                new = path[:i + 1] + [node] + path[i + 2:]
                if self._valid(new):
                    nbs.append(new)
        return nbs

    def _rand_neighbor(self, path, s, g):
        nbs = self._gen_neighbors(path, s, g)
        return random.choice(nbs) if nbs else None

    # ═══════════════════════════════════════════
    # Comparar todos
    # ═══════════════════════════════════════════
    def compare_all(self, start="ORIGEN", goal="DESTINO"):
        print("\n" + "=" * 70)
        print("  COMPARACION DE TODOS LOS ALGORITMOS")
        print("=" * 70)

        algos = [
            ("BFS", lambda: self.bfs(start, goal)),
            ("DFS", lambda: self.dfs(start, goal)),
            ("ILDFS", lambda: self.ildfs(start, goal)),
            ("Voraz", lambda: self.greedy(start, goal)),
            ("A*", lambda: self.a_star(start, goal)),
            ("Tabu", lambda: self.tabu_search(start, goal)),
            ("Recocido", lambda: self.simulated_annealing(
                start, goal)),
        ]

        results = {}
        for name, fn in algos:
            t0 = time.time()
            path = fn()
            el = time.time() - t0
            results[name] = {
                'path': path,
                'distance': (self.path_dist(path)
                             if path else float('inf')),
                'nodes': len(path) if path else 0,
                'time': el,
                'edges': list(self.explored_edges),
            }

        print("\n" + "=" * 80)
        print(f"  TABLA COMPARATIVA "
              f"(directa: {self.gb.direct_distance:.2f} km)")
        print("=" * 80)
        print(f"  {'Algo':<12} {'Dist':>10} {'Saltos':>7} "
              f"{'Tiempo':>9}  Camino")
        print(f"  {'─' * 12} {'─' * 10} {'─' * 7} "
              f"{'─' * 9}  {'─' * 35}")

        for n in sorted(results,
                        key=lambda k: results[k]['distance']):
            d = results[n]
            if d['path']:
                print(f"  {n:<12} {d['distance']:>8.2f}km "
                      f"{d['nodes'] - 1:>5}   "
                      f"{d['time']:>7.3f}s  "
                      f"{' -> '.join(d['path'])}")
            else:
                print(f"  {n:<12} {'N/A':>10} {'--':>7} "
                      f"{d['time']:>7.3f}s  Sin sol")

        # Mostrar que rutas tomaron
        print(f"\n  RUTAS DISTINTAS ENCONTRADAS:")
        seen = set()
        for n, d in results.items():
            if d['path']:
                key = tuple(d['path'])
                if key not in seen:
                    seen.add(key)
                    print(f"    {' -> '.join(d['path'])} "
                          f"({d['distance']:.2f} km) "
                          f"<- {n}")

        return results