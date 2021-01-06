import os.path as osp 
import os 
import glob

sceneFile = 'forcedFloor.txt' 
with open(sceneFile, 'r') as fIn:
    sceneList = fIn.readlines()[1:] 
sceneList = [x.strip() for x in sceneList ] 

roots = ['/eccv20dataset/DatasetNew_test/mainDiffLight_xml/',  \
        '/eccv20dataset/DatasetNew_test/mainDiffLight_xml1/',  \
        '/eccv20dataset/DatasetNew_test/mainDiffMat_xml/', \
        '/eccv20dataset/DatasetNew_test/mainDiffMat_xml1/', \
        '/eccv20dataset/DatasetNew_test/main_xml/', \
        '/eccv20dataset/DatasetNew_test/main_xml1/']
for root in roots:
    for scene in sceneList:
        scene = osp.join(root, scene )   
        print(scene )
        copyDir = osp.join(scene, 'oldFile')
        os.system('mv %s %s' % (osp.join(copyDir, '*'), scene ) )

roots = ['./mainDiffLight_xml/',  \
        './mainDiffLight_xml1/',  \
        './mainDiffMat_xml/', \
        './mainDiffMat_xml1/', \
        './main_xml/', \
        './main_xml1/']
for root in roots:
    for scene in sceneList:
        scene = osp.join(root, scene )
        print(scene )
        copyDir = osp.join(scene, 'oldFile')
        os.system('mv %s %s' % (osp.join(copyDir, '*'), scene ) )
