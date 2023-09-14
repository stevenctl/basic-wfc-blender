import bpy
import bpy_types


class Vec3:
    co: tuple[float, float, float] = (0, 0, 0)

    def __init__(self, x: float, y: float, z: float):
        self.co = (x, y, z)

    @property
    def x(self):
        return self.co[0]

    @property
    def y(self):
        return self.co[1]

    @property
    def z(self):
        return self.co[2]

    def __copy__(self):
        return Vec3(self.x, self.y, self.z)

    def __getitem__(self, key):
        return self.co[key]

    def __mul__(self, other):
        if type(other) in [float, int]:
            return Vec3(*[co * other for co in self.co])
        if type(other) == Vec3:
            return Vec3(
                self.x * other.x,
                self.y * other.y,
                self.z * other.z,
            )
        raise TypeError("Vec3 only supports multiplication with scalar or Vec3")

    def __add__(self, other):
        return Vec3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other):
        return Vec3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __iter__(self):
        return self.co.__iter__()

    def __str__(self):
        return "Vec3(%s,%s,%s)" % self.co

    def __repr__(self):
        return str(self)

    def ints(self) -> tuple[int, int, int]:
        return int(self.x), int(self.y), int(self.z)

    # returns a new Vec3 with diff added to x, y, or z (indexed by coord_comp)
    def add_comp(self, coord_comp, diff):
        new_co = list(self.co)
        new_co[coord_comp] += diff
        return Vec3(*new_co)

    # ret false if any component is outside the range of a/b
    def in_range(self, a, b):
        for co_comp in self.co:
            if co_comp < a or co_comp >= b:
                return False
        return True

    def set_comp(self, axis, v):
        new_co = list(self.co)
        new_co[axis] = v
        return Vec3(*new_co)


def _copy_object_shallow(ob, parent, collection=bpy.context.scene.collection, suffix=".copy"):
    # copy ob
    copy = ob.copy()
    copy.name = ob.name + suffix
    copy.data = ob.data.copy()
    copy.animation_data_clear()
    copy.parent = parent
    copy.matrix_parent_inverse = ob.matrix_parent_inverse.copy()
    collection.objects.link(copy)
    return copy


def copy_object(orig,
                collection=bpy.context.scene.collection,
                suffix=".copy",
                levels=5,
                exclude: [str] = None,
                root_parent=None):
    def recurse(ob, parent, depth):
        if depth > levels:
            return None
        children = [c for c in ob.children]
        if not parent:
            parent = root_parent
        copy = _copy_object_shallow(ob, parent, collection, suffix)
        for child in children:
            if exclude:
                if len([1 for exclusion in exclude if exclusion.lower() in child.name.lower()]) > 0:
                    continue
            recurse(child, copy, depth + 1)
        return copy
    return recurse(orig, root_parent, 0)


def apply_modifiers(obj: bpy.types.Object):
    objs = [obj]
    while len(objs) > 0:
        obj = objs.pop()
        for c in obj.children:
            objs.append(c)
        ctx = bpy.context.copy()
        ctx['object'] = obj
        for m in obj.modifiers:
            try:
                bpy.ops.object.modifier_apply(ctx, modifier=m.name)
            except RuntimeError:
                print(f"Error applying {m.name} to {obj.name}, removing it instead.")
                obj.modifiers.remove(m)

        for m in obj.modifiers:
            obj.modifiers.remove(m)


def save_debug_snapshot(suffix=""):
    import pathlib
    current_path = bpy.data.filepath
    if not current_path:
        print("cant debug from unsaved source")
        return

    current_path = pathlib.Path(current_path).absolute().parent
    debug_path = current_path / f"debug{suffix}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(debug_path))
    print(f"SNAPSHOT: {debug_path}")


def extract_children(parent: bpy.types.Object, child_filter=None):
    check = [c for c in parent.children]
    matches = []
    while len(check) > 0:
        child = check.pop()
        if not child_filter or child_filter(child):
            matches.append(child)
        for nested in child.children:
            check.append(nested)
    return matches


def delete_collection(existing: bpy.types.Collection):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in existing.objects:
        obj.select_set(True)
    bpy.ops.object.delete()
    bpy.data.collections.remove(existing)


def delete_object(existing: bpy.types.Object):
    bpy.ops.object.select_all(action='DESELECT')
    existing.select_set(True)
    bpy.ops.object.delete()


def get_or_create_collection(
        name: str,
        parent: bpy.types.Collection = None,
        replace=False):
    if not parent:
        parent = bpy.context.collection
    existing = parent.children.get(name)
    if existing:
        if replace:
            delete_collection(existing)
        else:
            return existing

    bpy.data.collections.new(name)
    collection = bpy.data.collections.get(name)
    parent.children.link(collection)
    return collection


def create_mesh(
        name: str,
        location: (float, float, float),
        verts: list[Vec3],
        edges: list[int],
        faces,
):
    view_layer = bpy.context.view_layer

    mesh_data = bpy.data.meshes.new(name.lower())
    mesh_data.from_pydata([v.co for v in verts], edges, faces)
    mesh_data.update()

    # Create new object with our light datablock.
    mesh_obj = bpy.data.objects.new(name=name, object_data=mesh_data)

    # Link light object to the active collection of current view layer,
    # so that it'll appear in the current scene.
    view_layer.active_layer_collection.collection.objects.link(mesh_obj)

    # Place light to a specified location.
    mesh_obj.location = location

    # And finally select it and make it active.
    mesh_obj.select_set(True)
    view_layer.objects.active = mesh_obj


def apply_transforms(ob, location=False, rotation=False, scale=False):
    if ob.data.users <= 1:
        bpy.ops.object.select_all(action="DESELECT")
        ob.select_set(True)
        bpy.ops.object.transform_apply(
            location=location, rotation=rotation, scale=scale, properties=False
        )
    else:
        if ob.scale[0] != ob.scale[1] or ob.scale[0] != ob.scale[2]:
            print(f"Warning: non-uniform scale on multi-user object {ob.name}.")
        if ob.rotation_euler[0] != ob.rotation_euler[1] or ob.rotation_euler[0] != ob.rotation_euler[2]:
            print(f"Warning: non-uniform scale on multi-user object {ob.name}.")


def flip_normals(ob):
    for obob in _flatten_children(ob):
        obob.data.flip_normals()


def _flatten_children(o):
    out = []
    work = [o]
    while len(work) > 0:
        c = work.pop()
        out.append(c)
        work += [ch for ch in c.children]

    return out
