import glob
import os.path as osp
import os

roots = ['main_xml', 'main_xml1', 'mainDiffMat_xml', 'mainDiffMat_xml1', 'mainDiffLight_xml', 'mainDiffLight_xml1']
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*') )
    for scene in scenes:
        print(scene )
        os.system('rm %s' % osp.join(scene, 'immatPart*.dat') )

