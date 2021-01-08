"""
batch process OR for 
- image
- layout bbox
- object bbox
- cam R, t
"""

from pathlib import Path
import random
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from tqdm import tqdm

from utils_OR import loadHdr, loadImage, scaleHdr
# from utils.utils_rui import vis_cube_plt, vis_axis
from utils_OR_mesh import load_OR_mesh, remove_top_down_faces, v_pairs_from_v2d_e, minimum_bounding_rectangle, mesh_to_contour, mesh_to_skeleton, transform_v, v_pairs_from_v3d_e, v_xytuple_from_v2d_e

from utils_OR_xml import get_XML_root, parse_XML_for_intrinsics
from utils_OR import in_frame, draw_projected_bdb3d
from PIL import Image, ImageDraw, ImageFont
from utils_OR_cam import normalize, read_cam_params, project_v

from utils_OR_xml import parse_XML_for_shapes
from utils_OR_mesh import loadMesh, computeBox, computeTransform
from utils_OR_transform import *
from utils_OR_mesh import writeMesh

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils.utils_rui import Arrow3D, vis_axis_xyz

dest_path = Path('/newfoundland2/ruizhu/siggraphasia20dataset/layout_labels')

rendering_path = Path('/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/DatasetCreation/main_xml')
layout_path = '/newfoundland2/ruizhu/siggraphasia20dataset/layoutMesh'
xml_path = '/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/scenes/xml'
root_uv_mapped = Path('/newfoundland2/ruizhu/siggraphasia20dataset/uv_mapped')
root_layoutMesh = Path('/newfoundland2/ruizhu/siggraphasia20dataset/layoutMesh')

scene_list = [f.name for f in rendering_path.iterdir() if (f.is_dir() and 'scene' in f.name)]
valid_scene_list = []
print('-- %d scenes found!'%len(scene_list))

for scene_name in scene_list:
    scene_rendering_path = rendering_path / scene_name
    frame_names_list = [f.name for f in scene_rendering_path.iterdir() if ('im_' in f.name and '.hdr' in f.name)]
    if len(frame_names_list) == 0:
        print('== No rendering found for scene %s; skipped.'%scene_name)
        continue
    valid_scene_list.append(scene_name)
    dest_scene_path = dest_path / scene_name
    dest_scene_path.mkdir(exist_ok=True)

    # ============= read layout =============
    if_debug_scene = random.random() < 1
    layout_obj_file = Path(layout_path) / scene_name / 'uv_mapped.obj'

    mesh = load_OR_mesh(layout_obj_file)
    mesh = remove_top_down_faces(mesh)
    v = np.array(mesh.vertices)
    e = mesh.edges

    # %matplotlib widget
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_proj_type('ortho')
    # ax.set_aspect("auto")
    # vis_axis(ax)
    # v_pairs = v_pairs_from_v3d_e(v, e)
    # for v_pair in v_pairs:
    #     ax.plot3D(v_pair[0], v_pair[1], v_pair[2])

    # find 2d floor contour
    v_2d, e_2d = mesh_to_contour(mesh)
    if if_debug_scene:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_aspect("equal")
        v_pairs = v_pairs_from_v2d_e(v_2d, e_2d)
        for v_pair in v_pairs:
            ax.plot(v_pair[0], v_pair[1])

    # finding minimum 2d cuboid from contour
    layout_hull_2d = minimum_bounding_rectangle(v_2d)
    hull_pair_idxes = [[0, 1], [1, 2], [2, 3], [3, 0]]
    hull_v_pairs = [([layout_hull_2d[idx[0]][0], layout_hull_2d[idx[1]][0]], [layout_hull_2d[idx[0]][1], layout_hull_2d[idx[1]][1]]) for idx in hull_pair_idxes]
    if if_debug_scene:
        for v_pair in hull_v_pairs:
            ax.plot(v_pair[0], v_pair[1], 'b--')
        plt.grid()
        layout_contour_2d_debug_path = dest_scene_path / 'layout_contour_2d_debug.png'
        plt.savefig(str(layout_contour_2d_debug_path))

    # simply mesh -> skeleton
    v_skeleton, e_skeleton = mesh_to_skeleton(mesh)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_proj_type('ortho')
    # ax.set_aspect("auto")
    # vis_axis(ax)
    # v_pairs = v_pairs_from_v3d_e(v_skeleton, e_skeleton)
    # for v_pair in v_pairs:
    #     ax.plot3D(v_pair[0], v_pair[1], v_pair[2])

    # 2d cuvboid hull -> 3d cuboid
    room_height = 3.
    layout_box_3d = np.hstack((np.vstack((layout_hull_2d, layout_hull_2d)), np.vstack((np.zeros((4, 1)), np.zeros((4, 1))+room_height))))    
    # vis_cube_plt(layout_box_3d, ax, 'b', linestyle='--')

    # transfer layout to world coordinates
    transformFile = Path(xml_path) / scene_name / 'transform.dat'
    # load transformations # writeShapeToXML.py L588
    with open(str(transformFile), 'rb') as fIn:
        transforms = pickle.load(fIn )
    transforms_layout = transforms[0]
    layout_box_3d_transform = transform_v(layout_box_3d, transforms_layout)

    v_skeleton_transform = transform_v(v_skeleton, transforms_layout)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_proj_type('ortho')
    # ax.set_aspect("auto")
    v_pairs = v_pairs_from_v3d_e(v_skeleton_transform, e_skeleton)
    # for v_pair in v_pairs:
    #     ax.plot3D(v_pair[0], v_pair[1], v_pair[2])
    # ax.view_init(elev=-71, azim=-65)

    # ============= Read cam intrinsics =============
    cam_file = Path(xml_path) / scene_name / 'cam.txt'
    main_xml_file = Path(xml_path) / scene_name / 'main.xml'
    root = get_XML_root(main_xml_file)
    cam_K, intrinsics = parse_XML_for_intrinsics(root)

    # ============= Read cam extrinsics =============
    cam_file = Path(xml_path) / scene_name / 'cam.txt'
    cam_params = read_cam_params(cam_file)

    # ============= load object bboxes =============
    shape_list = parse_XML_for_shapes(root, root_uv_mapped)
    # draw layout, cam and world coordinates in 3D
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_proj_type('ortho')
    # ax.set_aspect("auto")
    # v_pairs = v_pairs_from_v3d_e(v_skeleton_transform, e_skeleton)
    # for v_pair in v_pairs:
    #     ax.plot3D(v_pair[0], v_pair[1], v_pair[2])
    # ax.view_init(elev=-36, azim=89)
    # vis_axis(ax)

    vertices_list = []
    bverts_list = []
    faces_list = []
    num_vertices = 0
    obj_paths_list = []

    for shape_idx, shape in tqdm(enumerate(shape_list)):
        if 'container' in shape['filename']:
            continue

        if 'uv_mapped' in shape['filename']:
            obj_path = root_uv_mapped / shape['filename'].replace('../../../../../uv_mapped/', '')
        if 'layoutMesh' in shape['filename']:
            obj_path = root_layoutMesh / shape['filename'].replace('../../../../../layoutMesh/', '')

        vertices, faces = loadMesh(obj_path) # based on L430 of adjustObjectPoseCorrectChairs.py
        bverts, bfaces = computeBox(vertices )
        
        bverts_transformed, transforms_converted_list = transform_with_transforms_xml_list(shape['transforms_list'], bverts)
        vertices_transformed, _ = transform_with_transforms_xml_list(shape['transforms_list'], vertices)
        
        # if not(any(ext in shape['filename'] for ext in ['window', 'door', 'lamp'])):
        #     if 'scene' not in shape['filename']:
        #         vis_cube_plt(bverts_transformed, ax, 'r')
        
        vertices_list.append(vertices_transformed)
        faces_list.append(faces+num_vertices)
        num_vertices += vertices.shape[0]
        bverts_list.append(bverts_transformed)
        obj_paths_list.append(str(obj_path))
        
    scene_dict = {}
    scene_dict.update({'v_skeleton': v_skeleton_transform, 'e_skeleton': e_skeleton, 'layout_bbox_3d': layout_box_3d_transform})
    scene_dict.update({'obj_bboxes_3d_list': bverts_list, 'obj_paths_list': obj_paths_list})
    with open(str(dest_scene_path / 'scene_dict.pickle'),"wb") as f:
        pickle.dump(scene_dict, f)

    if if_debug_scene:
        # write to obj
        vertices_combine = np.vstack(vertices_list)
        faces_combine = np.vstack(faces_list)
        scene_mesh_debug_path = dest_scene_path / 'scene_mesh_debug.obj'
        writeMesh(scene_mesh_debug_path, vertices_combine, faces_combine)

    # ==== read frames
    frame_names_list = sorted(frame_names_list, key=lambda e: int(e.split('.')[0].split('_')[1]))
    for frame_id, frame_name in enumerate(frame_names_list):
        assert str(frame_id+1) in frame_name, '%dth frame has mismatch name of %s, from %s!'%(frame_id+1, frame_name, scene_rendering_path)
        frame_hdr_path = scene_rendering_path / frame_name

        if_debug_frame = random.random() < 1
        if_debug_frame = if_debug_frame or if_debug_scene

        im_width, im_height = 640, 480
        if if_debug_frame:
            im_hdr = loadHdr(frame_hdr_path)
            assert im_hdr.shape == (im_height, im_width, 3)
            # fig = plt.figure()
            # ax = fig.gca()
            # ax.set_aspect("equal")
            # plt.imshow(im_hdr)
            # plt.show()
            seg_file = str(frame_hdr_path).replace('im_', 'immask_').replace('hdr', 'png')
            seg = 0.5 * (loadImage(seg_file) + 1)[0, :, :]
            im, scale = scaleHdr(im_hdr, seg[:, :, np.newaxis])
            # im = im_hdr
            im_not_hdr = np.clip(im**(1.0/2.2), 0., 1.)
            im_uint8 = (255. * im_not_hdr).astype(np.uint8)
            frame_png_path = dest_scene_path / str(frame_hdr_path.name).replace('.hdr', '.png')
            print(frame_png_path)
            Image.fromarray(im_uint8).save(str(frame_png_path))
            # fig = plt.figure()
            # ax = fig.gca()
            # ax.set_aspect("equal")
            # plt.imshow(im_uint8)
            # plt.show()

        # ============= Read cam extrinsics =============
        cam_param = cam_params[frame_id]
        origin, lookat, up = np.split(cam_param.T, 3, axis=1)
        at_vector = lookat - origin
        # print(np.abs(np.dot(at_vector.flatten(), up.flatten())))
        assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

        zaxis = normalize(lookat - origin)
        # xaxis = normalize(np.cross(up.flatten(), zaxis.flatten()).reshape(3, 1))
        xaxis = normalize(np.cross(up.T, zaxis.T).T)
        yaxis = np.cross(zaxis.T, xaxis.T).T
        R_c = np.hstack([xaxis, yaxis, zaxis])
        R_c = np.linalg.inv(R_c)
        t_c = - R_c @ origin
        
        cam_dict = intrinsics
        cam_dict.update({'R_c': R_c, 't_c': t_c})
        cam_dict_path = dest_scene_path / ('cam_dict_%d.pickle'%(frame_id+1))
        with open(cam_dict_path,"wb") as f:
            pickle.dump(cam_dict, f)

        if if_debug_frame:
            v_proj = project_v(v_skeleton_transform, R_c, t_c, cam_K)
            img_map = Image.fromarray(im_uint8)
            width, height = img_map.size
            draw = ImageDraw.Draw(img_map)
            
            v_tuples = v_xytuple_from_v2d_e(v_proj, e_skeleton) # [(x1y1, x2y2), ...]
            v_tuples = [x for x in v_tuples if in_frame(x[0], width, height) or in_frame(x[1], width, height)]

            for v_tuple in v_tuples:
                draw.line([tuple(v_tuple[0]), tuple(v_tuple[1])], width=2)

            layout_box_3d_transform_proj, front_flags = project_v(layout_box_3d_transform, R_c, t_c, cam_K, if_only_proj_front_v=False, if_return_front_flags=True)
            draw_projected_bdb3d(draw, layout_box_3d_transform_proj, front_flags=front_flags, color=(255, 0, 0))

            layout_reproj_2d_debug_path = dest_scene_path / ('layout_reproj_2d_debug_%d.png'%(frame_id+1))
            img_map.save(str(layout_reproj_2d_debug_path))


        # ============= project objects and check visibility =============
        if if_debug_frame:    
            fig = plt.figure()
            plt.imshow(im_uint8)

        obj_vis_flags_list = []
        for bverts, obj_path in zip(bverts_list, obj_paths_list):
            # figure out obj visibility in the image
            valid_vertices = 0
            if_is_obj = 'alignedNew.obj' in str(obj_path)
            obj_box_3d_transform_proj, front_flags = project_v(bverts, R_c, t_c, cam_K, if_only_proj_front_v=False, if_return_front_flags=True)
            for idx in range(len(front_flags)):
                if front_flags[idx] and in_frame(obj_box_3d_transform_proj[idx], width, height):
                    valid_vertices += 1
            if valid_vertices > 0 and if_is_obj:
                bbox_color='g'
                obj_vis_flags_list.append(True)
            else:
                bbox_color='y'
                obj_vis_flags_list.append(False)

            if if_debug_frame and if_is_obj:
                v_list = obj_box_3d_transform_proj
                for idx_list in [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
                    for i in range(len(idx_list)-1):
                        if front_flags[idx_list[i]] and front_flags[idx_list[i+1]]:
                            plt.plot([v_list[idx_list[i]][0], v_list[idx_list[i+1]][0]], [v_list[idx_list[i]][1], v_list[idx_list[i+1]][1]], color=bbox_color)

        if if_debug_frame:    
            im_height, im_width = im_uint8.shape[:2]                    
            plt.xlim([-im_width, im_width*2])
            plt.ylim([im_height*2, -im_height])
            obj_reproj_2d_debug_path = dest_scene_path / ('obj_reproj_2d_debug_%d.png'%(frame_id+1))
            plt.savefig(str(obj_reproj_2d_debug_path))
        
        frame_dict = {}
        frame_dict.update({'layout_bbox_3d_proj': layout_box_3d_transform_proj, 'obj_box_3d_proj': obj_box_3d_transform_proj, 'obj_vis_flags_list': obj_vis_flags_list})
        with open(str(dest_scene_path / ('frame_dict_%d.pickle'%(frame_id+1))),"wb") as f:
            pickle.dump(frame_dict, f)

            
        # vis_cube_plt(layout_box_3d_transform, ax, 'b', '--')
        # vis_axis_xyz(ax, xaxis.flatten(), yaxis.flatten(), zaxis.flatten(), origin.flatten(), suffix='_c')
        # a = Arrow3D([origin[0][0], lookat[0][0]*2-origin[0][0]], [origin[1][0], lookat[1][0]*2-origin[1][0]], [origin[2][0], lookat[2][0]*2-origin[2][0]], mutation_scale=20,
        #                 lw=1, arrowstyle="->", color="k")
        # ax.add_artist(a)
        # a_up = Arrow3D([origin[0][0], origin[0][0]+up[0][0]], [origin[1][0], origin[1][0]+up[1][0]], [origin[2][0], origin[2][0]+up[2][0]], mutation_scale=20,
        #                 lw=1, arrowstyle="->", color="r")
        # ax.add_artist(a_up)


print('== %d/%d scenes are valid (has valid renderings)'%(len(valid_scene_list), len(scene_list)))