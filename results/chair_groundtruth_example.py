import bpy
from mathutils import Vector
from blender_shading import polygen_shader
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([-0.09764725714921951, -0.04728902131319046, -0.17917746305465698]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([-0.09764725714921951, 0.05893642082810402, -0.17917746305465698]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([-0.09764725714921951, 0.11737854778766632, -0.5344808101654053]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([-0.09764725714921951, 0.11737854778766632, -0.44175946712493896]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([-0.09764725714921951, 0.2887640595436096, -0.534480631351471]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([-0.09764725714921951, 0.2887640595436096, -0.441759318113327]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([0.11064518243074417, -0.04728902131319046, -0.17917746305465698]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([0.11064518243074417, 0.058936309069395065, -0.17917746305465698]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([0.11064518243074417, 0.11737854778766632, -0.534480631351471]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([0.11064518243074417, 0.11737854778766632, -0.44175946712493896]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([0.11064518243074417, 0.2887640595436096, -0.534480631351471]))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.019999999552965164, enter_editmode=False, location=Vector([0.11064518243074417, 0.2887640595436096, -0.44175946712493896]))
bpy.context.scene.objects[-1].select_set(True)
bpy.context.view_layer.objects.active = bpy.context.scene.objects[-1]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.join()
mat = bpy.data.materials.new('PKHG')
mat.use_nodes = True
polygen_shader(mat.node_tree, (0.9635047912597656, 0.4565802216529846, 0.7869526743888855, 1), 0.3878069818019867)
obj = bpy.context.active_object
if obj.data.materials:
	for i in range(len(obj.data.materials)):
		obj.data.materials[i] = mat
else:
	obj.data.materials.append(mat)
