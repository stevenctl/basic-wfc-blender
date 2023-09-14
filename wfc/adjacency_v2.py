import decimal
import typing

import bpy
from decimal import Decimal, getcontext as deccontext
import json
import os

from wfc.prune_lines import optimize_2d_mesh
from bpy_util import (
    copy_object,
    get_or_create_collection,
    flip_normals,
    apply_modifiers,
)

TILE_SIZE = Decimal("1")
Collection = bpy.types.Collection

Vert2 = tuple[Decimal, Decimal]
Vert3 = tuple[Decimal, Decimal, Decimal]
Vert = typing.Union[Vert2, Vert3]


# TODO maybe just use to_mesh
def copy_and_apply_modifiers(source_tiles, tile_size=Decimal(TILE_SIZE)):
    copy_col = get_or_create_collection(
        source_tiles.name + ".applied",
        parent=bpy.context.view_layer.layer_collection.collection,
    )
    for o in copy_col.objects:
        o.select_set(True)
        bpy.ops.object.delete(confirm=True)

    offset = 0
    import re

    # filter duplicates (ending like `.001`)
    filter_exp = re.compile("\\.\\d+$")
    for src in source_tiles.objects:
        if src.type != 'MESH' or filter_exp.search(src.name):
            continue

        copied = copy_object(src, collection=copy_col, suffix=".applied")
        apply_modifiers(copied)
        copied.location = [offset, 0, 0]
        offset += tile_size * 2

        # Just add a flipped version of everything for safetey
        flip = copy_object(copied, collection=copy_col, suffix=".Flip")
        flip.scale = (-1, 1, 1)
        bpy.ops.object.select_all(action="DESELECT")
        flip.select_set(True)
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True, properties=False
        )
        flip_normals(flip)
        offset += tile_size * 2

    return copy_col


def do_rotate_3(rot, v: Vert3):
    x, y = do_rotate(rot, v)
    _, _, z = v

    return x, y, z


def do_rotate(rot, v: Vert):
    """
    rotate x and y of a vert around (0,0)
    :param rot: degrees to rotate by
    :param v: vertex
    :param width:
    :return:
    """
    rot = rot % 360
    x, y = v[:2]
    if rot == 0:
        return x, y
    if rot == 90:
        return -y, x
    if rot == 180:
        return -x, -y
    if rot == 270:
        return y, -x

    raise Exception("do_rotate: Only 90 degree rotations are supported")


def do_flip(vert: Vert):
    x = vert[0]
    copy = [c for c in vert]
    copy[0] = -x
    return tuple(copy)


def detect_interiors(o):
    # first item is axis (0 = x, 1 = y, 2 = z), second is dir on axis
    axis_for_face = {
        "west": (0, -1),
        "east": (0, 1),
        "south": (1, -1),
        "north": (1, 1),
        "bottom": (2, -1),
        "top": (2, 1),
    }

    interiors = set()
    for face, item in axis_for_face.items():
        axis, sign = item
        detected_sign = 0
        max_poly = -float("inf")
        for p in o.data.polygons:
            normal = Decimal("%.3f" % p.normal[axis])
            if normal == Decimal("0"):
                continue

            normal_sign = 1 if normal > 0 else -1
            for vert_idx in p.vertices:
                vert_value = sign * o.data.vertices[vert_idx].co[axis]
                if vert_value > max_poly:
                    detected_sign = normal_sign
                    max_poly = vert_value
        if detected_sign != sign:
            interiors.add(face)
    return interiors


def extract_faces(tiles: Collection, tile_size=Decimal(TILE_SIZE)):
    def checker(val, want):
        ok, dist = val == want, abs(val - want)
        if not ok and dist < Decimal(".001"):
            # TODO this warning relies on vert and src being set in some other scope.. love python NO RULES!!
            print(
                f"{vert} on {src.name} seems using fine precision check the mesh and rounding in script"
            )
        return ok

    half_tile = tile_size / Decimal("2")
    plus_or_minus_half_tile = [
        (-half_tile, -half_tile),
        (-half_tile, half_tile),
        (half_tile, -half_tile),
        (half_tile, half_tile),
    ]

    faces = [
        {
            "name": "south",
            "rotation": 0,
            "axis": 1,
            "dir": -1,
        },
        {
            "name": "east",
            "rotation": 270,
            "axis": 0,
            "dir": 1,
        },
        {
            "name": "north",
            "rotation": 180,
            "axis": 1,
            "dir": 1,
        },
        {
            "name": "west",
            "rotation": 90,
            "axis": 0,
            "dir": -1,
        },
        {
            "name": "top",
            "vertical": True,
            "axis": 2,
            "dir": 1,
        },
        {
            "name": "bottom",
            "vertical": True,
            "axis": 2,
            "dir": -1,
        },
    ]
    faces_by_name = {}
    for f in faces:
        faces_by_name[f["name"]] = f
    src_objs = [o for o in tiles.objects if type(o.data) == bpy.types.Mesh]
    sockets = {}
    vsockets = {}

    face_debug = get_or_create_collection("face_debug", tiles, replace=True)

    for obj_idx, src in enumerate(src_objs):
        if type(src.data) != bpy.types.Mesh:
            print(f"Warning {src.name} is not a Mesh")
            continue

        # spread objects out with TILE_SIZE spacing in the middle
        src.location[0] = obj_idx * TILE_SIZE * 2
        src.location[1] = -24

        interior_faces = None
        try:
            interior_faces = detect_interiors(src)
        except Exception as e:
            print(e)

        i = Decimal(0)
        for face in faces:
            if interior_faces is None:
                interior = "unknown"
            else:
                interior = face["name"] in interior_faces
            i += 1
            vertical = "vertical" in face and face["vertical"]

            # Copy the object to create a "debug mesh" for each of the faces
            o: bpy.types.Object = copy_object(src, face_debug)
            o.name = src.name + ".face." + face["name"]
            o.location[1] += float(tile_size * i * 2)
            mesh: bpy.types.Mesh = o.data

            # Remove custom properties from this copy
            for k in [k for k in o.data.keys()]:
                if k.startswith("sock"):
                    del o.data[k]

            # Copy verts sitting on the correct face into the debug mesh
            verts = []
            vert_idx_map = {}
            for v in mesh.vertices:
                # TODO this round could bite in ass, is it even necessary with Decimal?
                round_fmt = "%." + str(decimal.getcontext().prec) + "f"
                vert = tuple([Decimal(round_fmt % n) for n in v.co])
                ok = vert[face["axis"]] == face["dir"] * half_tile
                if not ok:
                    continue
                vert_idx_map[v.index] = len(verts)

                if vertical:
                    z = vert[2]
                    if "offset" in face:
                        z += face["offset"]
                    verts.append((vert[0], vert[1], z))
                else:
                    # the rotation guarantees y always = 0 in the verts we're collecting here
                    # sockets just care about x and z
                    x, y = do_rotate(face["rotation"], vert)
                    verts.append((x, y, vert[2]))

            # Collect edge data so we can create debug mesh
            edges = []
            for e in mesh.edges:
                a, b = e.vertices[0], e.vertices[1]
                if a in vert_idx_map and b in vert_idx_map:
                    edges.append((vert_idx_map[a], vert_idx_map[b]))

            # Detect faces that sit on the tile boundary. It's hard to tell which edges/verts are part of the socket.
            # So we don't allow it.
            for f in mesh.polygons:
                all = True
                for li in f.loop_indices:
                    l = mesh.loops[li]
                    if l.vertex_index not in vert_idx_map:
                        all = False
                        break
                if all:
                    # Visual indication of incompatible tiles (mark the src mesh red)
                    src.color = [1, 0, 0, 1]
                    break

            opt_ig = 2 if vertical else 1  # 1 cuz y is always 0 for h faces
            verts, edges = optimize_2d_mesh(verts, edges, ignore=opt_ig)
            mesh.clear_geometry()
            mesh.from_pydata(verts, edges, [])

            if vertical:
                socket_names, new_socket = vertical_socket_from_verts(verts, vsockets)
                rot = 0
                for socket_name in socket_names:
                    if (
                        socket_name == "empty" or socket_name == "v_empty"
                    ) and interior == "unknown":
                        raise Exception(
                            "TODO a v face (%s) in %s requiring empty cannot detect normals"
                            % (face["name"], src.name)
                        )

                    if socket_name == "v_empty" and interior:
                        socket_name = "v_empty_interior"

                    src.data[f"socket_{face['name']}_{rot}"] = socket_name
                    o.data[f"socket_{rot}"] = socket_name
                    rot += 90
            else:
                socket_name, new_socket = side_socket_from_verts(verts, sockets)
                if (
                    socket_name == "empty" or socket_name == "v_empty"
                ) and interior == "unknown":
                    raise Exception(
                        "TODO a face (%s) in  %s requiring empty cannot detect normals"
                        % (face["name"], src.name)
                    )

                if socket_name == "empty" and interior:
                    socket_name = "empty_interior_s"
                src.data["socket_" + face["name"]] = socket_name
                o.data["socket"] = socket_name
            if new_socket:
                print(
                    f"New socket {socket_names if vertical else socket_name} for {o.name}"
                )

        # collect the verts that sit on corners
        mesh: bpy.types.Mesh = src.data
        wall_floor_verts = set()
        for v in mesh.vertices:
            # TODO this round could bite in ass, is it even necessary with Decimal?
            round_fmt = "%." + str(decimal.getcontext().prec) + "f"
            vert = tuple([Decimal(round_fmt % n) for n in v.co])
            n_half_tile = 0
            for vert_comp in vert:
                if abs(vert_comp) == half_tile:
                    n_half_tile += 1
            if n_half_tile >= 2:
                wall_floor_verts.add(vert)

        # if a mesh has all verts with all combos of +/- half tile (ignoring the component below) it matches
        wall_floor_ignored_index = [
            "long",  # x is zero (lateral)
            "lat",  # y is zero (longitudinal)
            "floor_ceil",  # z is zero (flat)
        ]

        center_tile_type = ""
        multiple_types = False
        for i in range(len(wall_floor_ignored_index)):
            k = wall_floor_ignored_index[i]
            found = 0
            for want in plus_or_minus_half_tile:
                # TODO optimizable.. but who cares
                # search all the verts, ignoring the component at `i`
                for got3 in wall_floor_verts:
                    got2 = got3[0:i] + got3[i + 1:3]
                    if want == got2:
                        found += 1
                        break
            if found == 4:
                if center_tile_type != "":
                    multiple_types = True
                center_tile_type = k
        src.data["center_tile_type"] = center_tile_type
        if multiple_types:
            print(f"WARNING: {src.name} has multiple center_tile_type")

    return sockets, vsockets


def side_socket_from_verts(verts, sockets):
    # Case 0: Empty socket is special
    if len(verts) == 0:
        return "empty", False

    verts_2d = extract_verts(verts)
    sock_hash = hash_verts(verts_2d)

    # Case 1: We've seen this before
    if sock_hash in sockets:
        return sockets[sock_hash], False

    # Case 2: this is a symmetry of something we've seen before
    flipped_verts = [do_flip(v) for v in verts_2d]
    flip_hash = hash_verts(flipped_verts)
    if flip_hash in sockets:
        return sockets[flip_hash] + "f", False

    # Case 3: This is new
    socket_name = f"sock_{len(sockets)}"

    # Special Case 4: This is symmetrical with itself
    if flip_hash == sock_hash:
        socket_name += "s"

    if socket_name in ["sock_0s", "sock_5s"]:
        print(f"{socket_name}: {flip_hash}")

    sockets[sock_hash] = socket_name
    return socket_name, True


def vertical_socket_from_verts(verts, sockets):
    verts_2d = extract_verts(verts, [0, 1])
    if len(verts) == 0:
        return ["v_empty" for i in range(4)], False

    # TODO can't assume mul of 4, if we have 180 degree or 360 degree symmetry there will be 2 or 1
    sock_name_pre = f"v{len(sockets)}"

    symmetrical = False
    for rot in [0, 90]:
        rotated_verts = [do_rotate(rot, v) for v in verts_2d]
        flipped_verts = [do_flip(v) for v in rotated_verts]
        if hash_verts(flipped_verts) == hash_verts(rotated_verts):
            symmetrical = True
            break

    out = []
    created_new, reused_existing = False, False
    for rot in [0, 90, 180, 270]:
        rotated_verts = [do_rotate(rot, v) for v in verts_2d]
        sock_hash = hash_verts(rotated_verts)
        sock_name = sock_name_pre + "_r" + str(rot)
        if symmetrical:
            sock_name += "s"

        if sock_hash in sockets:
            if created_new:
                print(f"Reusing {sockets[sock_hash]} instead of adding {sock_name}")
            reused_existing = True
            out.append(sockets[sock_hash])
        else:
            if reused_existing:
                print(
                    f"Creating new sock {sock_name} after previously reusing in a rotation"
                )
            created_new = True
            sockets[sock_hash] = sock_name
            out.append(sock_name)

    return out, created_new


def extract_verts(verts, indices=None):
    if indices is None:  # by default, get rid of y
        indices = [0, 2]

    return [[v[i] for i in indices] for v in verts]


# decimal will cause 1.00 or 1.000 - always use the same decimals in str
hash_fmt = {
    1: "%.3f",
    2: "%.3f,%.3f",
    3: "%.3f,%.3f,%.3f",
}


def hash_verts(verts):
    verts = [fix_vert(vert) for vert in verts]
    sorted_verts = sorted([hash_fmt[len(val)] % val for val in verts])
    return ";".join(sorted_verts)


# negative zero breaks string equality
def fix_vert(vert: tuple[Decimal]):
    return tuple([Decimal("0") if c == Decimal("0") else c for c in vert])


def generate_prototypes(objects: bpy.types.Collection):
    prototypes = [
        {
            "name": "empty",
            "mesh": "",
            "rotation": 0,
            "north": "empty;empty_only",
            "east": "empty;empty_only",
            "south": "empty;empty_only",
            "west": "empty;empty_only",
            "top": "v_empty;v_empty_only",
            "bottom": "v_empty;v_empty_only",
        },
        {
            "name": "empty_interior",
            "mesh": "",
            "rotation": 0,
            "north": "empty_interior_s",
            "east": "empty_interior_s",
            "south": "empty_interior_s",
            "west": "empty_interior_s",
            "top": "v_empty_interior",
            "bottom": "v_empty_interior",
        },
    ]
    for o in objects.objects:
        for rot in [0, 90, 180, 270]:
            try:
                prototypes.append(generate_prototype(o, rot))
            except Exception as e:
                print("-------")
                print(o.name)
                sock_keys = [k for k in o.data.keys() if k.startswith("socket_")]
                print(sock_keys)
                print("-------")
                raise e

    return prototypes


def flip_f(s):
    if s == "empty" or s.endswith("s"):
        return s
    if s.endswith("f"):
        return s[:-1]
    else:
        return s + "f"


def generate_prototype(object: bpy.types.Object, rotation: int):
    side_sockets = [
        object.data["socket_north"],
        object.data["socket_east"],
        object.data["socket_south"],
        object.data["socket_west"],
    ]

    # counter-clockwise rotation
    delta = int(rotation / 90)
    side_sockets = side_sockets[delta:] + side_sockets[:delta]

    # TODO this may still be needed.. probably is
    # if rotation >= 180:
    #     side_sockets = [flip_f(s) for s in side_sockets]

    # this is a vertical wall tile; need to rotate it (change lat <-> long) for 90/270
    center_tile_type = (
        "" if "center_tile_type" not in object.data else object.data["center_tile_type"]
    )
    if center_tile_type != "" and center_tile_type != "floor_ceil":
        if rotation % 180 != 0:
            center_tile_type = "lat" if center_tile_type == "long" else "long"

    return {
        "name": f"{object.name}_{rotation}",
        "mesh": f"{object.name}",
        "rotation": int(rotation),
        "north": side_sockets[0],
        "east": side_sockets[1],
        "south": side_sockets[2],
        "west": side_sockets[3],
        "top": object.data[f"socket_top_{rotation}"],
        "bottom": object.data[f"socket_bottom_{rotation}"],
        "center_tile_type": center_tile_type,
    }


def write_json(path, value):
    path = os.path.expanduser(path)
    with open(path, "w") as f:
        f.write(json.dumps(value, indent=2))


def main(source_tiles, out_path="~/prototypes.json"):
    if bpy.context and bpy.context.view_layer:
        ctx = bpy.context
        ctx.view_layer.objects.active = None
        bpy.ops.object.select_all(action="DESELECT")
    deccontext().prec = 3

    copies = copy_and_apply_modifiers(source_tiles)
    sockets, vsockets = extract_faces(copies)
    print(json.dumps(sockets, indent="  "))
    print(json.dumps(vsockets, indent="  "))
    prototypes = generate_prototypes(copies)

    if out_path != "":
        write_json(out_path, {"prototypes": prototypes})
        print(f"Wrote prototypes file to {out_path}.")


if __name__ == "__main__":
    col_name = "Base Tiles"
    source_tiles = bpy.data.collections.get(col_name)
    if not source_tiles:
        raise Exception("Could not find collection: {}".format(col_name))
    main()
