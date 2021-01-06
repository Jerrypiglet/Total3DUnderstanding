import glob
import os 
import os.path as osp 

roots = ['mainDiffLight_xml', 'mainDiffLight_xml1', \
        'mainDiffMat_xml', 'mainDiffMat_xml1']
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*') )
    for scene in scenes:
        print(scene )
        os.system('rm %s' % osp.join(scene, 'imsemLabel_*.npy') )
        os.system('rm %s' % osp.join(scene, 'imsemLabel2_*.npy') )

roots = ['main_xml', 'main_xml1']
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*') )
    for scene in scenes:
        print(scene )
        os.system('rm %s' % osp.join(scene, 'imsemLabel2_*.npy') ) 
