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

