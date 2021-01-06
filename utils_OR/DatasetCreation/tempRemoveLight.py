import glob
import os.path as osp 
import os 

roots = ['mainDiffMat_xml', 'mainDiffMat_xml1']
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*') )
    for scene in scenes: 
        print(scene )
        if osp.isdir(scene ):
            os.system('rm %s' % (osp.join(scene, 'imenvDirect_*.hdr') ) )
            os.system('rm %s' % (osp.join(scene, 'imshadingDirect_*.rgbe') ) )
