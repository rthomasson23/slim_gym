# load meshes
import trimesh
import numpy as np
import os
import sys


def load_meshes(path):
    # load meshes
    mesh = trimesh.load_mesh(path)
    return mesh

def convex_hull(mesh):
    # compute convex hull
    convex_hull = mesh.convex_hull
    return convex_hull

def save_convex_hull(convex_hull, path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    directory = os.path.dirname(path)
    convex_hull.export(directory + '/' + name + '_convex_hull.stl')
    print('Convex hull saved as ' + directory + '/' + name + '_convex_hull.stl')


if __name__=='__main__':

    path = '/home/aleberna/anaconda3/envs/slim/lib/python3.8/site-packages/robosuite/models/assets/grippers/meshes/sslim_hand/thumb_distal.stl'
   
    mesh = load_meshes(path)
    convex_hull = convex_hull(mesh)

    # visualize
    convex_hull.show()

    save = input('Do you want to save the convex hull? (y/n)')

    if save == 'y':
        save_convex_hull(convex_hull, path)
    else:
        sys.exit()
