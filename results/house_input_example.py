import os
import sys
sys.path.append(os.path.abspath("/home/baldur/Nextcloud_private/Bachelorarbeit/BlenderSkripts"))
from houseUtil import *

resetAll()

matBase = bpy.data.materials.new("PKHG")
matBase.use_nodes = True
setUnlitMaterial(1.0, 1.0, 1.0, matBase.node_tree)
matRoof = bpy.data.materials.new("PKHG")
matRoof.use_nodes = True
setUnlitMaterial(1.0, 0.0, 0.0, matRoof.node_tree)
matGable = bpy.data.materials.new("PKHG")
matGable.use_nodes = True
setUnlitMaterial(0.0, 1.0, 1.0, matGable.node_tree)
matChimney = bpy.data.materials.new("PKHG")
matChimney.use_nodes = True
setUnlitMaterial(1.0, 0.0, 1.0, matChimney.node_tree)
matWindow = bpy.data.materials.new("PKHG")
matWindow.use_nodes = True
setUnlitMaterial(0.0, 0.0, 1.0, matWindow.node_tree)
matDoor = bpy.data.materials.new("PKHG")
matDoor.use_nodes = True
setUnlitMaterial(1.0, 1.0, 0.0, matDoor.node_tree)

matBaseSlot = 0
matRoofSlot = 1
matGableSlot = 2
matChimneySlot = 3
matDoorSlot = 4
matWindowSlot = 5

house, roofHeight = createBaseHouse(11.15859603881836, 9.529009819030762, 10.222624778747559, 22.33353614807129)
doorCuboid = createCuboid(1.106482744216919, 1, 2.248027801513672)
doorCuboid.location = Vector([4.265504837036133, 1.9989993572235107, 1.1250139474868774])
addBoolMod(house, doorCuboid, "UNION", True)

selectSingleObject(house)
bpy.ops.object.mode_set(mode='EDIT')
house.active_material_index = matDoorSlot
bpy.context.object.active_material.shadow_method = 'NONE'
bpy.ops.object.material_slot_assign()
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')

windowCuboid0 = createCuboid(0.9528631567955017, 1, 1.8865307569503784)
windowCuboid0.location = Vector([4.265504837036133, 3.4612388610839844, 3.8682897090911865])
addBoolMod(house, windowCuboid0, "UNION", True)

windowCuboid1 = createCuboid(0.9898043274879456, 1, 0.9514836668968201)
windowCuboid1.location = Vector([4.265504837036133, 1.9716248512268066, 7.821371078491211])
addBoolMod(house, windowCuboid1, "UNION", True)

windowCuboid2 = createCuboid(0.9925244450569153, 1, 0.8081902265548706)
windowCuboid2.location = Vector([4.265504837036133, 1.8396790027618408, 8.902278900146484])
addBoolMod(house, windowCuboid2, "UNION", True)

windowCuboid3 = createCuboid(0.9348378777503967, 1, 0.7382674217224121)
windowCuboid3.location = Vector([4.265504837036133, 0.2569178342819214, 7.846984386444092])
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
camera.rotation_euler = Vector([1.5515564680099487, 1.3959285949383116e-09, 2.1704673767089844])
camera.location = Vector([29.3944149017334, 20.095600128173828, 9.254127502441406])
