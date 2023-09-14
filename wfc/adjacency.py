import os
import json
import math
import bpy
from decimal import (
    Decimal,
    getcontext as decimal_context,
)

from bpy_util import (
    get_or_create_collection,
    copy_object,
    flip_normals,
    apply_modifiers,
)
from wfc.prune_lines import optimize_2d_mesh

TILE_SIZE = 1
HALF_TILE = 0.5
TOLERANCE = 0.001
DECIMAL_PLACES = int(-math.log10(TOLERANCE))

FACES = [
    {"name": "south", "rotation": 0},
    {"name": "west", "rotation": 1},
    {"name": "north", "rotation": 2},
    {"name": "east", "rotation": 3},
    {"name": "top", "v_offset": -TILE_SIZE},
    {"name": "bottom"},
]


def round_dec(n):
    rounded = Decimal(f"%.{DECIMAL_PLACES}f" % n)
    return Decimal("0") if rounded.is_zero() else rounded


def detect_interiors(o):
    """
    Looks at the mesh from each each direction to determine if that face
    should be on the inside or the outside of our generated mesh
    """
    # first item is axis (0 = x, 1 = y, 2 = z), second is the sign/direction
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
            normal = Decimal(f"%.{DECIMAL_PLACES}f" % p.normal[axis])
            if normal.is_zero():
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


def extract_face(src_obj, face, face_collection):
    # make a copy we can modify safely
    face_name = face["name"]
    face_obj = copy_object(src_obj, face_collection, f".face.{face_name}")

    # rotate it and offset it to make it easier to filter verts
    face_obj.rotation_euler[2] = math.radians(90 * face.get("rotation", 0))
    face_obj.location = (0, 0, face.get("v_offset", 0))

    # apply the tranformation to the geometry
    bpy.ops.object.select_all(action="DESELECT")
    face_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, scale=False, rotation=True)

    # Horizontal faces check the Y (forward) axis, Vertical uses the Z axis
    check_axis = 2 if face_name in {"top", "bottom"} else 1

    # Filter vertices to those sitting on the south face
    verts = []
    index_map = {}
    for i, vert in enumerate(face_obj.data.vertices):
        if abs(-HALF_TILE - vert.co[check_axis]) > TOLERANCE:
            continue
        index_map[i] = len(verts)

        # round the components of the vert
        # the axis we just checked doesn't matter going forward
        rounded_vert = list(vert.co)
        rounded_vert[check_axis] = 0
        for axis in range(3):
            rounded_vert[axis] = round_dec(rounded_vert[axis])
        verts.append(tuple(rounded_vert))

    # restore edge data (indices change since we removed a lot of verts)
    # not totally necessary, but we can inspect this mesh for debugging
    edges = []
    for e in face_obj.data.edges:
        a, b = e.vertices[0], e.vertices[1]
        if a in index_map and b in index_map:
            edges.append((index_map[a], index_map[b]))

    # remove collinear vertices that don't affect the contour
    verts, edges = optimize_2d_mesh(verts, edges, ignore_axis=check_axis)

    face_obj.data.clear_geometry()
    face_obj.data.from_pydata(verts, edges, [])

    def make_2d(v):
        return tuple(list(v)[:check_axis] + list(v)[check_axis + 1 :])

    return face_obj, [make_2d(v) for v in verts]


def hash_verts(verts_2d):
    format = f"%.{DECIMAL_PLACES}f,%.{DECIMAL_PLACES}f"
    return ";".join(sorted([format % v for v in verts_2d]))


def rotate_vert(vert, k):
    x, y = vert
    match k % 4:
        case 0:
            return x, y
        case 1:
            return -y, x
        case 2:
            return -x, -y
        case 3:
            return y, -x


def h_socket(verts_2d, sockets):
    # Case 0: Empty socket is special
    if len(verts_2d) == 0:
        return "empty"

    sock_hash = hash_verts(verts_2d)

    # Case 1: We've seen this before
    if sock_hash in sockets:
        return sockets[sock_hash]

    # Case 2: this is a symmetry of something we've seen before
    flipped_verts = [(-v[0], v[1]) for v in verts_2d]
    flip_hash = hash_verts(flipped_verts)
    if flip_hash in sockets:
        return sockets[flip_hash] + "f"

    # Case 3: This is new
    socket_name = f"sock_{len(sockets)}"

    # Special Case 4: This is symmetrical with itself
    if flip_hash == sock_hash:
        socket_name += "s"

    sockets[sock_hash] = socket_name
    return socket_name


def v_socket(verts_2d, sockets):
    # Case 0: Empty socket is special
    if len(verts_2d) == 0:
        return "v_empty"

    sock_hash = hash_verts(verts_2d)

    # This is new, generate rotated versions
    if sock_hash not in sockets:
        base_socket_name = f"v_sock_{len(sockets)}"
        for r in range(4):
            rotated_verts = [rotate_vert(v, r) for v in verts_2d]
            rotated_hash = hash_verts(rotated_verts)
            sockets[rotated_hash] = base_socket_name + f"_r{r}"

    return sockets[sock_hash]


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

    # filter duplicates (endings like `.001` generated by blender)
    filter_exp = re.compile("\\.\\d+$")
    for src in source_tiles.objects:
        if src.type != "MESH" or filter_exp.search(src.name):
            continue

        copied = copy_object(src, collection=copy_col, suffix=".applied")
        apply_modifiers(copied)
        copied.location = [offset, 0, 0]
        offset += tile_size * 2

        # Just add a flipped version of everything for safetey
        flip = copy_object(copied, collection=copy_col, suffix=".Flip")
        flip.location = [offset, 0, 0]
        flip.scale = (-1, 1, 1)
        bpy.ops.object.select_all(action="DESELECT")
        flip.select_set(True)
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True, properties=False
        )
        flip_normals(flip)
        offset += tile_size * 2

    return copy_col


def compute_sockets(tiles):
    """
    Custom properties named `sock_{face}` will be added
    to each object in the given collection of tiles.
    """
    decimal_context().prec = DECIMAL_PLACES
    face_col = get_or_create_collection("face_debug", tiles)

    h_sockets, v_sockets = {}, {}

    for tile in tiles.objects:
        interiors = detect_interiors(tile)
        for face_index, face in enumerate(FACES):
            debug_obj, verts_2d = extract_face(tile, face, face_col)

            # align the debug object with the tile's mesh
            debug_obj.location = (
                tile.location[0],
                TILE_SIZE * 2 * (face_index + 1),
                tile.location[1],
            )

            sock_name = "UNKNOWN"
            if face["name"] in {"top", "bottom"}:
                sock_name = v_socket(verts_2d, v_sockets)
            else:
                sock_name = h_socket(verts_2d, h_sockets)

            if "empty" in sock_name and face["name"] in interiors:
                sock_name += "_interior"
            tile.data[f"socket_{face['name']}"] = sock_name


def rotate_socket(socket: str, k):
    if socket.startswith("v_empty"):
        return socket
    if not socket.endswith(("r0", "r1", "r2", "r3")):
        raise Exception("Cannot rotate socket: ", socket)
    return socket[:-1] + str((int(socket[-1]) + k) % 4)


def generate_prototype(object: bpy.types.Object, rotation: int):
    """
    Reads the mesh data from a Blender object and creates
    a Prototype variant for each 90 degree rotation
    """
    side_sockets = [
        object.data["socket_north"],
        object.data["socket_east"],
        object.data["socket_south"],
        object.data["socket_west"],
    ]

    # counter-clockwise rotation
    side_sockets = side_sockets[rotation:] + side_sockets[:rotation]

    return {
        "name": f"{object.name}_{rotation}",
        "mesh": f"{object.name}",
        "rotation": int(rotation),
        "north": side_sockets[0],
        "east": side_sockets[1],
        "south": side_sockets[2],
        "west": side_sockets[3],
        "top": rotate_socket(object.data["socket_top"], rotation),
        "bottom": rotate_socket(object.data["socket_bottom"], rotation),
    }


def generate_prototypes(tiles):
    """
    Given a collection of objects with adjacency data
    returns a list of prototype structs to be used in WFC
    """
    protos = [
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
            "north": "empty_interior",
            "east": "empty_interior",
            "south": "empty_interior",
            "west": "empty_interior",
            "top": "v_empty_interior",
            "bottom": "v_empty_interior",
        },
    ]
    for o in tiles.objects:
        for r in range(4):
            protos.append(generate_prototype(o, r))
    return protos


def compute_adjacency(source_tiles, out_path="~/prototypes.json"):
    """
    Duplicates the source tiles and applies modifiers
    to be used as the tileset for WFC.

    These duplicated tiles will have custom properties with their
    adjacency data, and a JSON file will be written describing the
    complete set of prototypes.
    """
    tiles = copy_and_apply_modifiers(bpy.context.scene.wfc_source_tiles)
    compute_sockets(tiles)
    prototypes = generate_prototypes(tiles)

    if out_path != "":
        path = os.path.expanduser(out_path)
        with open(path, "w") as f:
            f.write(json.dumps({"prototypes": prototypes}, indent=2))
        print(f"Wrote prototypes file to {out_path}.")
