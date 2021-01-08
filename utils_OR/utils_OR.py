import os.path as osp
import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

def loadHdr(imName, if_resize=False, imWidth=None, imHeight=None, if_channel_first=False):
    if not(osp.isfile(imName ) ):
        print(imName )
        assert(False )
    im = cv2.imread(imName, -1)
    # print(imName, im.shape, im.dtype)

    if im is None:
        print(imName )
        assert(False )

    if if_resize:
        im = cv2.resize(im, (imWidth, imHeight), interpolation = cv2.INTER_AREA )
    im = im[:, :, ::-1]
    if if_channel_first:
        im = np.transpose(im, [2, 0, 1])
    return im

def in_frame(p, width, height):
    if p[0]>0 and p[0]<width and p[1]>0 and p[1]<height:
        return True
    else:
        return False

# from utils.utils_rui import clip
# def clip2rec(polygon, W, H, line_width=5):
#     # if not fix_polygon:
#     #     return polygon
#     if all_outside_rect(polygon, W, H):
#         return []
#     rectangle = [(-line_width, -line_width), (W+line_width, -line_width), (W+line_width, H+line_width), (-line_width, H+line_width)]
#     return clip(polygon, rectangle)

# def all_outside_rect(polygon, W, H):
#     if all([x[0] < 0 or x[0] >= W or x[1] < 0 or x[1] >= H for x in polygon]):
#         return True
#     else:
#         return False

def draw_lines_notalloutside_image(draw, v_list, idx_list, front_flags, color=(255, 255, 255), width=5):
    assert len(v_list) == len(front_flags)
    for i in range(len(idx_list)-1):
        if front_flags[idx_list[i]] and front_flags[idx_list[i+1]]:
            draw.line([v_list[idx_list[i]], v_list[idx_list[i+1]]], width=width, fill=color)

def draw_projected_bdb3d(draw, bdb2D_from_3D, front_flags=None, color=(255, 255, 255), width=5):
    bdb2D_from_3D = [tuple(item) for item in bdb2D_from_3D]
    if front_flags is None:
        front_flags = [True] * len(bdb2D_from_3D)
    assert len(front_flags) == len(bdb2D_from_3D)

    for idx_list in [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        draw_lines_notalloutside_image(draw, bdb2D_from_3D, idx_list, front_flags, color=color, width=width)

    # W, H = img_map.size

    # print(clip2rec([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[2], bdb2D_from_3D[3], bdb2D_from_3D[0]], W=W, H=H, line_width=width))

    # draw.line(clip2rec([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[2], bdb2D_from_3D[3], bdb2D_from_3D[0]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[4], bdb2D_from_3D[5], bdb2D_from_3D[6], bdb2D_from_3D[7], bdb2D_from_3D[4]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[0], bdb2D_from_3D[4]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[1], bdb2D_from_3D[5]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[2], bdb2D_from_3D[6]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[3], bdb2D_from_3D[7]], W=W, H=H, line_width=width),
    #     fill=color, width=width)