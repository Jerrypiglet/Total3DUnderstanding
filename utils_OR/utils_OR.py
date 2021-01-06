import os.path as osp
import cv2
import numpy as np
import trimesh

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

def draw_projected_bdb3d(im, v, e, cam_K, if_save = True, save_path=''):
        from PIL import Image, ImageDraw, ImageFont

        img_map = Image.fromarray(self.img_map[:])

        draw = ImageDraw.Draw(img_map)

        width = 5

        if type == 'prediction':
            boxes = self.pre_boxes
            cam_R = self.pre_cam_R
        else:
            boxes = self.gt_boxes
            cam_R = self.gt_cam_R

        for coeffs, centroid, class_id, basis in zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis']):
            if class_id not in RECON_3D_CLS:
                continue
            center_from_3D, invalid_ids = proj_from_point_to_2d(centroid, self.cam_K, cam_R)
            bdb3d_corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            bdb2D_from_3D = proj_from_point_to_2d(bdb3d_corners, self.cam_K, cam_R)[0]

            # bdb2D_from_3D = np.round(bdb2D_from_3D).astype('int32')
            bdb2D_from_3D = [tuple(item) for item in bdb2D_from_3D]

            color = nyu_color_palette[class_id]

            draw.line([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[2], bdb2D_from_3D[3], bdb2D_from_3D[0]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[4], bdb2D_from_3D[5], bdb2D_from_3D[6], bdb2D_from_3D[7], bdb2D_from_3D[4]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[0], bdb2D_from_3D[4]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[1], bdb2D_from_3D[5]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[2], bdb2D_from_3D[6]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[3], bdb2D_from_3D[7]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)

            draw.text(tuple(center_from_3D), NYU40CLASSES[class_id],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20))

        img_map.show()

