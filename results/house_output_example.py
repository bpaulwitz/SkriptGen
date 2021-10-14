import os
import sys
sys.path.append(os.path.abspath("/home/baldur/Nextcloud_private/Bachelorarbeit/BlenderSkripts"))
from houseUtil import *

resetAll()

matBase = bpy.data.materials.new("PKHG")
matBase.use_nodes = True
setUnlitMaterial(0.4712221622467041, 0.47328057885169983, -0.7576540112495422, matBase.node_tree)
matRoof = bpy.data.materials.new("PKHG")
matRoof.use_nodes = True
setUnlitMaterial(-0.05468820035457611, 1.305660367012024, -0.40902185440063477, matRoof.node_tree)
matGable = bpy.data.materials.new("PKHG")
matGable.use_nodes = True
setUnlitMaterial(-0.4273459315299988, 1.5827051401138306, 0.8011043071746826, matGable.node_tree)
matChimney = bpy.data.materials.new("PKHG")
matChimney.use_nodes = True
setUnlitMaterial(-2.0830230712890625, 0.33635905385017395, -1.8597520589828491, matChimney.node_tree)
matWindow = bpy.data.materials.new("PKHG")
matWindow.use_nodes = True
setUnlitMaterial(2.6010854244232178, 0.47474178671836853, -0.8205788731575012, matWindow.node_tree)
matDoor = bpy.data.materials.new("PKHG")
matDoor.use_nodes = True
setUnlitMaterial(4.484918594360352, 8.095175743103027, 9.273391723632812, matDoor.node_tree)

matBaseSlot = 0
matRoofSlot = 1
matGableSlot = 2
matChimneySlot = 3
matDoorSlot = 4
matWindowSlot = 5

house, roofHeight = createBaseHouse(15.784584045410156, 14.054458618164062, 9.215445518493652, 8.923402786254883)
doorCuboid = createCuboid(3.6691176891326904, 1, 1.1876258850097656)
doorCuboid.location = Vector([6.208758354187012, 3.800753116607666, 6.6192522048950195])
addBoolMod(house, doorCuboid, "UNION", True)

selectSingleObject(house)
bpy.ops.object.mode_set(mode='EDIT')
house.active_material_index = matDoorSlot
bpy.context.object.active_material.shadow_method = 'NONE'
bpy.ops.object.material_slot_assign()
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')

windowCuboid0 = createCuboid(1.6810246706008911, 1, -0.08844684064388275)
windowCuboid0.location = Vector([-0.4275173246860504, -0.5330144166946411, 2.3082337379455566])
addBoolMod(house, windowCuboid0, "UNION", True)

selectSingleObject(house)
bpy.ops.object.mode_set(mode='EDIT')
house.active_material_index = matDoorSlot
bpy.context.object.active_material.shadow_method = 'NONE'
bpy.ops.object.material_slot_assign()
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')

windowCuboid3 = createCuboid(1.81694495677948, 1, -1.2529634237289429)
windowCuboid3.location = Vector([1.786787986755371, -0.2999981641769409, 2.0385959148406982])
addBoolMod(house, windowCuboid3, "UNION", True)

selectSingleObject(house)
bpy.ops.object.mode_set(mode='EDIT')
house.active_material_index = matWindowSlot
bpy.context.object.active_material.shadow_method = 'NONE'
bpy.ops.object.material_slot_assign()
bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.object.camera_add()
camera = bpy.context.active_object
bpy.context.scene.camera = camera
scene = bpy.context.scene
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.resolution_percentage = 100
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.color_mode = 'RGBA'
camera.rotation_euler = Vector([-1.6084895133972168, -3.3314971923828125, -4.339078426361084])
camera.location = Vector([-4.442261219024658, -0.6372851133346558, 0.2956205904483795])
