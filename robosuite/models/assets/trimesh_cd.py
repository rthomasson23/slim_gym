import trimesh 
import os

mesh = trimesh.load("/home/rthom/Documents/Research/TRI/bdml_mujoco/robosuite/models/assets/robots/panda_wrist/meshes/forearm_right_old.stl")
convex_decomposition = trimesh.decomposition.convex_decomposition(mesh, maxhullcount=3, findBestPlane=True)
new_mesh = []
for ii in convex_decomposition:
    new_part = trimesh.Trimesh(**ii, process=True)
    new_mesh.append(new_part)

for i, m in enumerate(new_mesh):
    m.visual.vertex_colors = trimesh.visual.random_color()
trimesh.Scene(new_mesh).show()