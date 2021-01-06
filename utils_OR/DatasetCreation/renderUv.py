import glob
import os
import os.path as osp
import argparse
import xml.etree.ElementTree as et
from xml.dom import minidom

def transformToXml(root ):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString= xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0 ]
    xmlString = '\n'.join(xmlString )
    return xmlString

parser = argparse.ArgumentParser()
# Directories
parser.add_argument('--xmlRoot', default="/siggraphasia20dataset/code/Routine/scenes/xml", help="outdir of xml file")
# Start and end point
parser.add_argument('--rs', default=0, type=int, help='the width of the image' )
parser.add_argument('--re', default=5, type=int, help='the height of the image' )
# xml file
parser.add_argument('--xmlFile', default='main', help='the xml file')
# output file
parser.add_argument('--outRoot', default='/siggraphasia20dataset/code/Routine/DatasetCreation', help='output directory')
# Render Mode
parser.add_argument('--mode', default=7, type=int, help='the information being rendered')
# Control
parser.add_argument('--forceOutput', action='store_true', help='whether to overwrite previous results')
# Program
parser.add_argument('--program', default='/siggraphasia20dataset/OptixRendererUVcoord/src/bin/optixRenderer', help='the location of render' )
opt = parser.parse_args()


scenes = glob.glob(osp.join(opt.xmlRoot, 'scene*') )
scenes = [x for x in scenes if osp.isdir(x) ]
scenes = sorted(scenes )
for n in range(opt.rs, min(opt.re, len(scenes ) ) ):
    scene = scenes[n]
    sceneId = scene.split('/')[-1]

    print('%d/%d: %s' % (n, len(scenes), sceneId ) )

    outDir = osp.join(opt.outRoot, opt.xmlFile + '_' + opt.xmlRoot.split('/')[-1], sceneId )
    if not osp.isdir(outDir ):
        continue

    xmlFile = osp.join(scene, '%s.xml' % opt.xmlFile )
    camFile = osp.join(scene, 'cam.txt' )
    if not osp.isfile(xmlFile ) or not osp.isfile(camFile ):
        continue

    cmd = '%s -f %s -c %s -o %s -m %d' % (opt.program, xmlFile, 'cam.txt', osp.join(outDir, 'im.hdr'), opt.mode )
    print(cmd )

    if opt.forceOutput:
        cmd += ' --forceOutput'

    os.system(cmd )
