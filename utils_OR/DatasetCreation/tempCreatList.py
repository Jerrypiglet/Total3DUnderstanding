import glob
import os.path as osp
import os

root = '../scenes/xml/'
sceneDirs = glob.glob(osp.join(root, 'scene*') )
sceneIds = [x.split('/')[-1] for x in sceneDirs ]

listName = 'forcedFloor.txt'
with open(listName, 'w') as fOut:
    fOut.write('# Forced floor\n')
    for n in range(0, len(sceneDirs ) ):
        sceneDir = sceneDirs[n]
        sceneId = sceneIds[n]

        if osp.isdir(osp.join(sceneDir, 'oldFile') ):
            fOut.write('%s\n' % sceneId )
