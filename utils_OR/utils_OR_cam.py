import numpy as np
import os.path as osp

def read_cam_params(camFile):
    assert osp.isfile(str(camFile))
    with open(str(camFile), 'r') as camIn:
    #     camNum = int(camIn.readline().strip() )
        cam_data = camIn.read().splitlines()
    cam_num = int(cam_data[0])
    cam_params = np.array([x.split(' ') for x in cam_data[1:]]).astype(np.float)
    assert cam_params.shape[0] == cam_num * 3
    cam_params = np.split(cam_params, cam_num, axis=0) # [[origin, lookat, up], ...]
    return cam_params

def normalize(x):
    return x / np.linalg.norm(x)

def project_v(v, cam_R, cam_t, cam_K, if_only_proj_front_v=True, if_return_front_flags=False):
    v_transformed = cam_R @ v.T + cam_t
#     print(v_transformed[2:3, :])
    if if_only_proj_front_v:
        v_transformed = v_transformed * (v_transformed[2:3, :] > 0.)
    p = cam_K @ v_transformed
    if not if_return_front_flags:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T
    else:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T, (v_transformed[2:3, :] > 0.).flatten().tolist()

def project_v_homo(v, cam_transformation4x4, cam_K):
    # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/img30.gif
    # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
    v_homo = np.hstack([v, np.ones((v.shape[0], 1))])
    cam_K_homo = np.hstack([cam_K, np.zeros((3, 1))])
#     v_transformed = cam_R @ v.T + cam_t

    v_transformed = cam_transformation4x4 @ v_homo.T
    v_transformed_nonhomo = np.vstack([v_transformed[0, :]/v_transformed[3, :], v_transformed[1, :]/v_transformed[3, :], v_transformed[2, :]/v_transformed[3, :]])
#     print(v_transformed.shape, v_transformed_nonhomo.shape)
    v_transformed = v_transformed * (v_transformed_nonhomo[2:3, :] > 0.)
    p = cam_K_homo @ v_transformed
    return np.vstack([p[0, :]/p[2, :], p[1, :]/p[2, :]]).T
