import bpy

bl_info = {
    "name": "WFC Tile Generator",
    "blender": (3, 5, 0),
    "category": "Collection",
}


class OperatorAdjacencyFromGeo(bpy.types.Operator):
    bl_idname = "wfc.geo_adjacency"
    bl_label = "Adjacency From Geometry"

    def execute(self, context):
        if not bpy.context.scene.wfc_source_tiles:
            self.report({'ERROR'}, "must set source tiles")
            return {'FINISHED'}
        try:
            from wfc.adjacency_v2 import main as adjacency
            adjacency(bpy.context.scene.wfc_source_tiles)
        except Exception as e:
            self.report({'ERROR'}, e)
        return {'FINISHED'}


class OperatorGenerateRandom(bpy.types.Operator):
    bl_idname = "wfc.generate_random"
    bl_label = "Generate Random"

    def execute(self, context):
        if not bpy.context.scene.wfc_source_tiles:
            self.report({'ERROR'}, "must set source tiles")
            return {'FINISHED'}
        try:
            from wfc.solver import test_with_retry
            test_with_retry()
        except Exception as e:
            self.report({'ERROR'}, e)
        return {'FINISHED'}


class WFCPanel(bpy.types.Panel):
    bl_label = "WFC Tile Generator"
    bl_category = "WFC"
    bl_idname = "COLLECTION_PT_wfc"

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    # bl_space_type = 'PROPERTIES'
    # bl_region_type = 'WINDOW'
    # bl_context = "collection"

    def draw(self, context):
        self.layout.prop(context.scene, "wfc_source_tiles")
        self.layout.operator(OperatorAdjacencyFromGeo.bl_idname)
        self.layout.operator(OperatorGenerateRandom.bl_idname)


def register():
    bpy.types.Scene.wfc_prototypes_path = \
        bpy.props.PointerProperty(type=bpy.types.Collection)
    bpy.types.Scene.wfc_source_tiles = \
        bpy.props.PointerProperty(type=bpy.types.Collection)

    bpy.utils.register_class(OperatorAdjacencyFromGeo)
    bpy.utils.register_class(OperatorGenerateRandom)
    bpy.utils.register_class(WFCPanel)


def unregister():
    bpy.utils.unregister_class(OperatorAdjacencyFromGeo)
    bpy.utils.unregister_class(OperatorGenerateRandom)
    bpy.utils.unregister_class(WFCPanel)


if __name__ == "__main__":
    # for local dev
    import os
    import sys
    sys.path.append(os.getcwd())
    register()
