import os.path as osp
import os
import glob

sceneFile = 'forcedFloor.txt'
with open(sceneFile, 'r') as fIn:
    sceneList = fIn.readlines()[1:]
sceneList = [x.strip() for x in sceneList ]

roots = ['/siggraphasia20dataset/code/Routine/scenes/xml',
        '/siggraphasia20dataset/code/Routine/scenes/xml1']
for root in roots:
    for scene in sceneList:
        scene = osp.join(root, scene )
        print(scene )
        copyDir = osp.join(scene, 'oldFile')
        #os.system('mkdir %s' % copyDir )
        #os.system('mv %s %s' % (osp.join(scene, '*'), copyDir ) )
        #os.system('cp %s %s' % (osp.join(copyDir, 'transform.dat'), scene ) )
        os.system('cp %s %s' % (osp.join(copyDir, 'cam.txt'), scene ) )
