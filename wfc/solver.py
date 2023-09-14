import json
import math
import typing
from random import randint
from os import path
from time import perf_counter_ns
from copy import deepcopy, copy

import bpy.types
from bpy.types import Collection
from bpy_util import get_or_create_collection

Vec3 = tuple[int, int, int]
TILE_SIZE = 1


def add_vec3(a: Vec3, b: Vec3):
    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    )


opposing_faces = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "top": "bottom",
    "bottom": "top",
}

face_deltas = {
    "north": (0, 1, 0),
    "south": (0, -1, 0),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
    "top": (0, 0, 1),
    "bottom": (0, 0, -1),
}

vertical = {"top", "bottom"}

horizontal = {"north", "south", "east", "west"}


class Prototype:
    name: str
    mesh: str
    rotation: int
    cube_id: int
    n_filled: int
    flip: bool

    north: str
    east: str
    south: str
    west: str
    top: str
    bottom: str

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "")
        self.mesh = kwargs.get("mesh", "")
        self.rotation = int(kwargs.get("rotation", "-1"))
        self.cube_id = int(kwargs.get("cube_id", "-1"))
        self.n_filled = int(kwargs.get("n_filled", "-1"))
        self.flip = bool(kwargs.get("flip ", False))
        self.north = kwargs.get("north", "")
        self.east = kwargs.get("east", "")
        self.south = kwargs.get("south", "")
        self.west = kwargs.get("west", "")
        self.top = kwargs.get("top", "")
        self.bottom = kwargs.get("bottom", "")

    def __eq__(self, other: "Prototype"):
        return self.name == other.name

    def json_dict(self):
        return {**self.__dict__}

    def sockets_for_face(self, face: str) -> set[str]:
        f = getattr(self, face)
        if not f:
            f = ""
        return set(f.split(";"))

    def is_empty(self, internal=False):
        s = {"empty", "v_empty"}
        if internal:
            s.add("empty_interior")
        return self.name in s

    def __str__(self):
        return self.name

    def copy(self):
        return deepcopy(self)


class Cell:
    coord: Vec3
    possibilities: list[Prototype]

    def __init__(self, coord: Vec3):
        self.coord = coord

    def __str__(self):
        x, y, z = self.coord
        entropy = self.entropy()
        value = self.possibilities[0] if entropy == 1 else f"entropy={entropy}"
        return f"({x}, {y}, {z} [{value}])"

    @staticmethod
    def _symmetrical(sock):
        return sock.endswith("s") or sock == "empty" or sock == "empty_only"

    def allowed_sockets(self, face: str) -> set[str]:
        out = set()
        for p in self.possibilities:
            for sock in p.sockets_for_face(face):
                # horizontal sockets only connect to their mirror
                # TODO special case for empty socket
                if face in horizontal:
                    if self._symmetrical(sock):
                        pass  # no need to flip; symmetrical
                    elif sock.endswith("f"):
                        sock = sock[:-1]
                    else:
                        sock += "f"

                out.add(sock)
        return out

    def solved(self) -> bool:
        return self.entropy() == 1

    def entropy(self) -> int:
        return len(self.possibilities)

    def collapse_random(self):
        idx = randint(0, len(self.possibilities) - 1)
        self.possibilities = [self.possibilities[idx]]

    def possibly_empty(self, internal=False) -> bool:
        for p in self.possibilities:
            if p.is_empty(internal):
                return True
        return False

    def constrain_not_name(self, name: str, if_not_zero=False):
        self.constrain_name(name, False, if_not_zero)

    def constrain_name(self, name: str, match: bool = True, if_not_zero=False):
        """

        :param name: name to match
        :param match: if False, match everything _but_ this name
        :param if_not_zero: cancel if leaves 0 possibilities
        :return:
        """
        old = len(self.possibilities)

        def check(a, b):
            if match:
                return a == b
            return a != b

        filtered = [p for p in self.possibilities if check(p.name, name)]
        if len(filtered) == 0 and if_not_zero:
            return False
        self.possibilities = filtered
        return old != len(self.possibilities)

    def constrain_to_prefix(self, prefix: str):
        old = len(self.possibilities)
        self.possibilities = [
            p for p in self.possibilities if p.name.startswith(prefix)
        ]
        return len(self.possibilities) != old

    def constrain_to_neighbor(self, cell: "Cell", face: str):
        """
        :param cell: the cell we want this cell to be compatible with
        :param face: the face to check for sockets on
        :return: whether the possibilities here changed and need propagation
        """

        def has_it():
            return (
                len([p for p in self.possibilities if "Corner.Diag.Vert" in p.name]) > 0
            )

        had_it = has_it()

        # TODO can infer face by looking at neighbor coords :)
        allowed_sockets = cell.allowed_sockets(face)
        # "butt to butt rule"
        if not cell.possibly_empty():
            if "empty" in allowed_sockets:
                allowed_sockets.remove("empty")
                allowed_sockets.add("empty_only")
            if "v_empty" in allowed_sockets:
                allowed_sockets.remove("v_empty")
                allowed_sockets.add("v_empty_only")

        my_face = opposing_faces[face]
        old = len(self.possibilities)
        self.possibilities = [
            p
            for p in self.possibilities
            if len(p.sockets_for_face(my_face).intersection(allowed_sockets)) > 0
        ]

        if len(self.possibilities) == 0:
            raise ImpossibleException(self)
        return old != len(self.possibilities)


class ImpossibleException(Exception):
    cell: Cell

    def __init__(self, cell: Cell, *args, **kwargs):
        super(ImpossibleException, self).__init__(*args, **kwargs)
        self.cell = cell


class Grid:
    grid: list[list[list[Cell]]]
    size: Vec3

    def __init__(self, size: Vec3):
        grid = []
        sx, sy, sz = size
        for z in range(sz):
            layer = []
            grid.append(layer)
            for y in range(sy):
                row = []
                layer.append(row)
                for x in range(sx):
                    row.append(Cell((x, y, z)))

        self.grid = grid
        self.size = size

    def get(self, coord: Vec3):
        x, y, z = coord
        return self.grid[z][y][x]

    def iterator(self):
        sx, sy, sz = self.size
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    yield self.get((x, y, z))

    def is_in_bounds(self, coord):
        for i in range(len(coord)):
            v = coord[i]
            if v < 0 or v >= self.size[i]:
                return False
        return True

    def copy(self) -> "Grid":
        return deepcopy(self)


class Solver:
    grid: Grid
    prototypes: list[Prototype]
    iteration: int = 0
    restarts: int = 0
    collapses: int = 0
    propagation_stack: list[Vec3] = []

    _empty_proto: Prototype = None
    _starting_grid: Grid

    def __init__(
        self,
        grid: Grid = None,
        prototypes: list[Prototype] = None,
        prototype_path: str = "",
    ):
        if not grid:
            grid = Grid((10, 10, 10))

        if not prototypes:
            if not prototype_path:
                prototype_path = path.expanduser("~/prototypes.json")
            with open(prototype_path, "r") as f:
                prototypes = json.load(f)["prototypes"]
                # TODO more validation when loading external file
                if type(prototypes) != list:
                    raise Exception(
                        f"loaded prototypes was not a list; got {type(prototypes)}"
                    )
                for i in range(len(prototypes)):
                    prototypes[i] = Prototype(**prototypes[i])

        self.grid = grid
        self.prototypes = copy(prototypes)
        self.init_grid()

    def init_grid(self):
        for cell in self.grid.iterator():
            cell.possibilities = self.prototypes

    def write_not_empty(self, coords: list[Vec3]):
        # First, list coord, and the 7 other coords, above, north and east
        filled_coords = set()
        for coord in coords:
            non_empty_coords = [
                coord,
                # inc one dir
                add_vec3(coord, (1, 0, 0)),
                add_vec3(coord, (0, 1, 0)),
                add_vec3(coord, (0, 0, 1)),
                # inc two dir
                add_vec3(coord, (1, 1, 0)),
                add_vec3(coord, (0, 1, 1)),
                add_vec3(coord, (1, 0, 1)),
                # inc all dir
                add_vec3(coord, (1, 1, 1)),
            ]

            # ignore out of bounds coords
            for write_coord in non_empty_coords:
                if not self.grid.is_in_bounds(write_coord):
                    continue
                filled_coords.add(write_coord)

        # Next, find coords that are totally surrounded by non-empty cells
        enclosed_coords = set()
        for filled in filled_coords:
            adjacent_non_empty = 0
            for delta in face_deltas.values():
                coord = add_vec3(filled, delta)
                if coord in filled_coords:
                    adjacent_non_empty += 1
            # handle enclosed cells
            if adjacent_non_empty == 6:
                enclosed_coords.add(filled)

        # Next, remove that from the "filled" coords
        for enclosed in enclosed_coords:
            filled_coords.remove(enclosed)

        # Finally, remove the empty possibilities; track what changed
        changed = set()
        for coord in filled_coords:
            changed_mt = self.grid.get(coord).constrain_not_name("empty")
            changed_int = self.grid.get(coord).constrain_not_name("empty_interior")
            if changed_mt or changed_int:
                changed.add(coord)
        for enclosed in enclosed_coords:
            # don't remove empty interior here, just regular empty
            if self.grid.get(enclosed).constrain_not_name("empty"):
                changed.add(enclosed)

        # Fill all cells that can have air, with air
        emptied = self.empty_all(exceptions=enclosed_coords)
        changed = changed.union(emptied)
        self.queue_propagation(changed)

    def empty_edges(self):
        if self.iteration != 0:
            print("warning: empty_edges called in the middle of solving")

        c = set()
        for cell in self.grid.iterator():
            x, y, z = cell.coord
            xs, ys, zs = self.grid.size
            x_invalid = 0 < x < xs - 1
            y_invalid = 0 < y < ys - 1
            z_invalid = z < zs - 1
            if x_invalid and y_invalid and z_invalid:
                continue
            cell.constrain_name("empty")
            # self.propagate(cell.coord)
            c.add(cell.coord)
        print(f"{len(c)} edges emptied")
        return c

    def empty_all(self, exceptions: set[Vec3] = None):
        if not exceptions:
            exceptions = set()
        if self.iteration != 0:
            print("warning: empty_all called in the middle of solving")
        changed = set()
        for cell in self.grid.iterator():
            if cell.coord in exceptions:
                continue

            if cell.constrain_name("empty", if_not_zero=True):
                changed.add(cell.coord)

        return changed

    def min_entropy(self) -> Cell:
        out: Cell = None
        for cell in self.grid.iterator():
            if cell.entropy() < 2:
                if cell.entropy() == 0:
                    # this specific exception shouldn't happen -
                    # it should have been caught during propagation
                    raise ImpossibleException(cell)
                continue  # don't include collapsed cells
            if not out or cell.entropy() < out.entropy():
                out = cell
        return out

    def restart(self):
        self.restarts += 1
        self.propagation_stack = []
        self.grid = self._starting_grid.copy()

    def solve(self, iterations, restart=True):
        for i in range(iterations):
            if self.iterate(restart=restart):
                break

    def iterate(self, restart=True):
        """
        collapse the cell with the lowest entropy and propagate
        :return:
        """
        if self.iteration == 0:
            self._starting_grid = self.grid.copy()
        self.iteration += 1

        if len(self.propagation_stack) == 0:
            self.collapses += 1
            cell = self.min_entropy()
            if not cell:
                return True
            cell.collapse_random()
            self.propagation_stack.append(cell.coord)
        try:
            self.propagate_step()
        except ImpossibleException as e:
            if restart:
                self.restart()
            else:
                raise e
        return False

    def propagate(self, coord):
        self.queue_propagation(coord)
        while self.propagate_step():
            pass

    def queue_propagation(self, coord: typing.Union[Vec3, set[Vec3]]):
        items = [coord]
        if coord is set:
            items = coord
        for i in items:
            if i not in self.propagation_stack:
                self.propagation_stack.append(i)

    def propagate_step(self):
        if len(self.propagation_stack) == 0:
            return False
        cur_coord = self.propagation_stack.pop()
        cur_cell = self.grid.get(cur_coord)
        for direction in opposing_faces.keys():
            next_coord = add_vec3(cur_coord, face_deltas[direction])
            if not self.grid.is_in_bounds(next_coord):
                continue
            next_cell = self.grid.get(next_coord)
            changed = next_cell.constrain_to_neighbor(cur_cell, direction)
            if changed and next_coord not in self.propagation_stack:
                self.propagation_stack.append(next_coord)
        return len(self.propagation_stack) > 0

    def print_stats(self):
        success = False
        try:
            self.min_entropy()
            success = True
        except ImpossibleException:
            pass

        print(
            f"""
        size: {self.grid.size}
        iterations: {self.iteration}
        restarts: {self.restarts}
        collapses: {self.collapses}
        success: {success}
        propagation stack size: {len(self.propagation_stack)}
        propagation stack head: {
            self.propagation_stack[-1]
            if len(self.propagation_stack) > 0 else None},
        """
        )


rotation_delta = {
    0: (0, 0),
    90: (1, 0),
    180: (1, 1),
    270: (0, 1),
}


class Renderer:
    tile_size: float
    parent_collection: Collection
    grid: Grid

    def __init__(
        self,
        grid: Grid,
        tile_size=TILE_SIZE,
        collection: bpy.types.Collection = None
    ):
        if not collection:
            collection = bpy.context.scene.collection

        self.grid = grid
        self.parent_collection = collection
        self.tile_size = tile_size

    def render(self, debug_cubes, render_empty=False):
        seen = set()

        collection = get_or_create_collection(
            "wfc.grid_render", self.parent_collection, replace=True
        )
        for cell in self.grid.iterator():
            if cell.coord in seen:
                print(f"wtf! {cell.coord}")
            seen.add(cell.coord)
            proto: Prototype = None
            mesh_name = "Unknown"
            if cell.entropy() == 1:
                proto = cell.possibilities[0]
                mesh_name = proto.mesh
                if proto.name == "empty" and not render_empty:
                    continue
            if mesh_name == "":
                mesh_name = "Unknown"

            if mesh_name == "Unknown" and not debug_cubes:
                continue

            mesh_obj = bpy.data.objects.get(mesh_name)
            if not mesh_obj:
                print(
                    f"warning: could not find object {mesh_name} referenced by cell {cell.coord}"
                )
                continue
            x, y, z = cell.coord
            render_name = f"wfc.grid_render.cell.{x}.{y}.{z}"
            render_obj = bpy.data.objects.new(
                render_name, object_data=mesh_obj.data.copy()
            )
            collection.objects.link(render_obj)
            if mesh_name == "Unknown":
                if cell.entropy() == 0:
                    render_obj.color = (1, 0, 0, 1)
                elif cell.solved() and cell.possibilities[0].name == "empty":
                    render_obj.color = (0, 1, 0, 1)
                elif cell.solved():
                    render_obj.color = (1, 0, 1, 1)
                elif not cell.possibly_empty(internal=True):
                    # cyan can't be empty (even internal)
                    render_obj.color = (0, 1, 1, 1)
                elif not cell.possibly_empty(internal=False):
                    # yellow can't be empty (but can be internal)
                    render_obj.color = (1, 1, 0, 1)
            render_obj = bpy.data.objects.get(render_name)
            if not render_obj:
                print(f"warning: did not not spawn object {render_name}")
                continue
            render_obj.location = (
                self.tile_size * x,
                self.tile_size * y,
                self.tile_size * z,
            )

            if proto:
                render_obj.rotation_euler = (0, 0, math.radians(proto.rotation))
                # delta = rotation_delta[proto.rotation]
                # render_obj.location[0] += delta[0] * self.tile_size
                # render_obj.location[1] += delta[1] * self.tile_size

            # Strip out debug data from the source mesh object
            # TODO figure out if there is a way to share mesh data but also change custom properties...
            sock_keys = [k for k in render_obj.data.keys() if k.startswith("socket_")]
            for k in sock_keys:
                del render_obj.data[k]

            render_obj.data["possible"] = ", ".join(
                [p.name for p in cell.possibilities]
            )
            render_obj.data["possible_n"] = str(len(cell.possibilities))
            # Add our own debug data
            for face in opposing_faces.keys():
                render_obj.data[f"allowed_{face}"] = ", ".join(
                    list(cell.allowed_sockets(face))
                )
                if cell.solved():
                    render_obj.data[f"socket_{face}"] = str(
                        cell.possibilities[0].sockets_for_face(face)
                    )


def test(iterations=1000, s=None):
    if not s:
        grid = Grid((10, 10, 10))
        s = Solver(grid)
    try:
        s.solve(iterations, restart=False)
    except ImpossibleException as e:
        print("Impossible at: ", e.cell)
        pass

    return s


def test_draw(iterations=1000, shape="4x3x2"):
    g = Grid((10, 10, 10))
    s = Solver(g)

    test_shapes = {
        "diag": {(1, 1, 1), (2, 2, 1)},
        "diagv": {(1, 1, 1), (2, 1, 2)},
        "slab": {
            (1, 1, 1),
            (2, 1, 1),
            (3, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (3, 2, 1),
            (1, 3, 1),
            (2, 3, 1),
            (3, 3, 1),
        },
        "o": {
            (1, 1, 1),
            (2, 1, 1),
            (3, 1, 1),
            (1, 2, 1),
            (3, 2, 1),
            (1, 3, 1),
            (2, 3, 1),
            (3, 3, 1),
        },
        "min": {(1, 1, 1)},
        "one": {
            (1, 1, 1),
            (2, 1, 1),
        },
        "L": [
            (1, 1, 2),
            (1, 1, 1),
            (2, 1, 1),
            (1, 4, 2),
            (1, 4, 1),
            (1, 5, 1),
            (2, 4, 2),
            (2, 4, 1),
            (2, 5, 1),
        ],
        "4x3x2": [
            (1, 1, 1),
            (2, 1, 1),
            (3, 1, 1),
            (4, 1, 1),
            (5, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (3, 2, 1),
            (4, 2, 1),
            (5, 2, 1),
            (1, 1, 2),
            (2, 1, 2),
            (3, 1, 2),
            (4, 1, 2),
            (5, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
            (3, 2, 2),
            (4, 2, 2),
            (5, 2, 2),
            (1, 1, 4),
            (2, 1, 4),
            (3, 1, 4),
            (4, 1, 4),
            (5, 1, 4),
            (1, 1, 4),
            (2, 1, 4),
            (3, 1, 4),
            (4, 1, 4),
            (5, 1, 4),
        ],
        "3x2x3": [
            (1, 1, 3),
            (2, 1, 3),
            (3, 1, 3),
            (1, 1, 2),
            (2, 1, 2),
            (3, 1, 2),
            (1, 1, 1),
            (2, 1, 1),
            (3, 1, 1),
        ],
        "3x3x3": [
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (1, 1, 2),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
        ],
    }

    try:
        s.write_not_empty(test_shapes[shape])
    except ImpossibleException as e:
        print("Impossible at: ", e.cell)
        pass

    return test(iterations, s=s)


def test_with_retry(iterations=1000, render=True, debug_cubes=False, mode="rand"):
    modes = {
        "rand": test,
        "draw": test_draw,
    }

    st = perf_counter_ns()
    s = modes[mode](iterations)
    print(f"solved in {(perf_counter_ns() - st) / 1000000}ms")

    # Debug Info (console)
    success = False
    try:
        s.min_entropy()
        success = True
    except ImpossibleException:
        pass
    s.print_stats()
    # Debug Info (3D)
    r = None
    if render:
        r = Renderer(s.grid)
        r.render(debug_cubes)
    return s, r


if __name__ == "__main__":
    test_with_retry(render=False, mode="draw")
