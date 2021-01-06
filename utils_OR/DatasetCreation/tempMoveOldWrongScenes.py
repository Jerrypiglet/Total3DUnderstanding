import glob
import os.path as osp
import os

roots = ['main_xml', 'main_xml1',  \
         'mainDiffMat_xml', 'mainDiffMat_xml1', \
         'mainDiffLight_xml', 'mainDiffLight_xml1']
dstRoot = '/siggraphasia20dataset/code/Routine/DatasetWrong/'

with open('forcedFloor.txt', 'r') as fIn:
    sceneList = fIn.readlines()[1:]
sceneList = [x.strip() for x in sceneList ]

for root in roots:
    dst = osp.join(dstRoot, root )
    os.system('mkdir %s' % dst )
    for scene in sceneList:
        os.system('mv %s %s' % (osp.join(root, scene ), dst ) )
