import glob
import os
import os.path as osp

roots = ['main_xml', 'main_xml1', 'mainDiffLight_xml', 'mainDiffLight_xml1',
         'mainDiffMat_xml', 'mainDiffMat_xml1']
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*_*') )
    cnt = 0
    for scene in scenes:
        imgs = glob.glob(osp.join(scene, 'im_*.hdr') )
        if len(imgs ) == 0:
            print(scene )
            os.system('rm -r %s' % scene )
            cnt +=1
    print(root, cnt )
