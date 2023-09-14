import copy


def collinear(p1, p2, p3, ignore=-1):
    v = [v for v in [p1, p2, p3]]
    if ignore >= 0:
        for i, vv in enumerate(v):
            vv = list(vv)
            vv.pop(ignore)
            v[i] = tuple(vv)

    x, y = v[0]
    m, n = v[1]
    a, b = v[2]
    return a*(n - y) + m*(y - b) + x*(b - n) == 0


def optimize_2d_mesh(verts, edges, ignore=-1):
    # immutable input
    verts = copy.deepcopy(verts)
    edges = copy.deepcopy(edges)

    # map of vert_idx -> []vert_idx that are connected to it
    edge_map = {}

    def map_edge(a, b):
        assert a != b

        for a, b in [(a, b), (b, a)]:
            if a not in edge_map:
                edge_map[a] = []
            edge_map[a].append(b)

    for a, b in edges:
        map_edge(a, b)

    assert len(edge_map) == len(verts)

    def offset(keep, remove):
        return keep if keep < remove else keep - 1

    def remove_vert(idx):
        verts.pop(idx)
        new_edge_map = {}
        for k, vl in edge_map.items():
            if k == idx:
                continue
            new_edge_map[offset(k, i)] = [offset(v, i) for v in vl if v != i]
        return new_edge_map

    i = 0
    while i < len(verts):
        # can only check collinear for a group of 3
        if i in edge_map and len(edge_map[i]) == 2:
            j, k = edge_map[i]
            if collinear(verts[i], verts[j], verts[k], ignore=ignore):
                edge_map = remove_vert(i)
                oj, ok = j, k
                j = j if j < i else j - 1
                k = k if k < i else k - 1
                try:
                    map_edge(j, k)
                except:
                    print(i, j, k, oj, ok)
                    raise
                continue

        i += 1

    edges = set()
    for a, neighbors in edge_map.items():
        for b in neighbors:
            edges.add(tuple(sorted([a, b])))

    return verts, list(edges)


def test():
    print(collinear(

        (-0.300, -0.300),
        (-0.300, 0.500),
        (-0.300, -0.500),
    ))
    print("-")
    VERTS = [
        (-0.300, -0.500, -0.300),
        (-0.300, -0.500, 0.500),
        (-0.300, -0.500, -0.500),
    ]

    EDGES = [(1, 0), (0, 2)]
    verts, edges = optimize_2d_mesh(VERTS, EDGES, ignore=1)
    print(VERTS)
    print(EDGES)
    print("to")
    print(verts)
    print(edges)


if __name__ == "__main__":
    test()
