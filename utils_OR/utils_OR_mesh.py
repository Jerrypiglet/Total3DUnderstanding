import trimesh
import numpy as np

def load_OR_mesh(layout_obj_file):
    mesh = trimesh.load_mesh(str(layout_obj_file))
    mesh = as_mesh(mesh)
    return mesh

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def remove_top_down_faces(mesh):
    v = np.array(mesh.vertices)
    f = list(np.array(mesh.faces))
    f_after = []
    for f0 in f:
        if not(v[f0[0]][2]==v[f0[1]][2]==v[f0[2]][2]):
            f_after.append(f0)
    new_mesh = trimesh.Trimesh(vertices=v, faces=np.asarray(f_after))
    return new_mesh

def mesh_to_contour(mesh):
    mesh = remove_top_down_faces(mesh)
    v = np.array(mesh.vertices)
    e = np.array(mesh.edges)

    v_new_id_list = []
    v_new_id = 0
    floor_z = np.amin(v[:, -1])
    for v0 in v:
        if v0[2]==floor_z:
            v_new_id_list.append(v_new_id)
            v_new_id += 1
        else:
            v_new_id_list.append(-1)
            
    v_new = np.array([v[x][:2] for x in range(len(v)) if v_new_id_list[x]!=-1])
    e_new = np.array([[v_new_id_list[e[x][0]], v_new_id_list[e[x][1]]] for x in range(len(e)) if (v_new_id_list[e[x][0]]!=-1 and v_new_id_list[e[x][1]]!=-1)])

    return v_new, e_new

def mesh_to_skeleton(mesh):
    mesh = remove_top_down_faces(mesh)
    v = np.array(mesh.vertices)
    e = mesh.edges

    floor_z = np.amin(v[:, -1])
    ceil_z = np.amax(v[:, -1])
    e_new = []
    for e0 in e:
        z0, z1 = v[e0[0]][2], v[e0[1]][2]
        if z0 == z1:
            e_new.append(e0)
        elif np.array_equal(v[e0[0]][:2], v[e0[1]][:2]):
            e_new.append(e0)
    e_new = np.array(e_new)

    return v, e_new

def v_pairs_from_v3d_e(v, e):
    v_pairs = [(np.array([v[e0[0]][0], v[e0[1]][0]]), np.array([v[e0[0]][1], v[e0[1]][1]]), np.array([v[e0[0]][2], v[e0[1]][2]])) for e0 in e]
    return v_pairs

def v_pairs_from_v2d_e(v, e):
    v_pairs = [(np.array([v[e0[0]][0], v[e0[1]][0]]), np.array([v[e0[0]][1], v[e0[1]][1]])) for e0 in e]
    return v_pairs

def v_xytuple_from_v2d_e(v, e):
    v_pairs = [(v[e0[0]], v[e0[1]]) for e0 in e]
    return v_pairs

def transform_v(vertices, transforms):
    assert transforms[0][0]=='s' and transforms[1][0]=='rot' and transforms[2][0]=='t'
    # following computeTransform()
    assert len(vertices.shape)==2
    assert vertices.shape[1]==3

    s = transforms[0][1]
    scale = np.array(s, dtype=np.float32 ).reshape(1, 3)
    vertices = vertices * scale
    rotMat = s = transforms[1][1]
    vertices = np.matmul(rotMat, vertices.transpose() )
    vertices = vertices.transpose()
    t = s = transforms[2][1]
    trans = np.array(t, dtype=np.float32 ).reshape(1, 3)
    vertices = vertices + trans
    
    return vertices