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
parser.add_argument('--rs', default=4, type=int, help='the width of the image' )
parser.add_argument('--re', default=5, type=int, help='the height of the image' )
# xml file
parser.add_argument('--xmlFile', default='main', help='the xml file')
# output file
parser.add_argument('--outRoot', default='/eccv20dataset/DatasetNew_test/', help='output directory')
# Render Mode
parser.add_argument('--mode', default=0, type=int, help='the information being rendered')
# Control
parser.add_argument('--forceOutput', action='store_true', help='whether to overwrite previous results')
parser.add_argument('--medianFilter', action='store_true', help='whether to use median filter')
# Program
parser.add_argument('--program', default='/siggraphasia20dataset/OptixRenderer/src/bin/optixRenderer', help='the location of render' )
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

    tree  = et.parse(xmlFile )
    root = tree.getroot()

    shapes = root.findall('shape')
    isFindAreaLight = False
    for shape in shapes:
        emitters = shape.findall('emitter')
        if len(emitters ) > 0:
            isFindAreaLight = True
            break

    sensor = root.findall('sensor')[0]
    film = sensor.findall('film')[0]
    integers = film.findall('integer')
    for integer in integers:
        if integer.get('name') == 'height':
            integer.set('value', '120')
        elif integer.get('name') == 'width':
            integer.set('value', '160')

    newXmlFile = xmlFile.replace('.xml', '_direct.xml')
    xmlString = transformToXml(root )

    if osp.isfile(newXmlFile ):
        print('%s already exists' % newXmlFile )
    else:
        with open(newXmlFile, 'w') as xmlOut:
            xmlOut.write(xmlString )

    cmd = '%s -f %s -c %s -o %s -m %d --maxPathLength 2 ' % (opt.program, newXmlFile, 'cam.txt', osp.join(outDir, 'imDirect.rgbe'), opt.mode )

    if opt.forceOutput:
        cmd += ' --forceOutput'

    if opt.medianFilter:
        cmd += ' --medianFilter'

    if not isFindAreaLight:
        print('Warning: no area light found, may need more samples.' )
        cmd += ' --maxIteration 5'
    else:
        cmd += ' --maxIteration 6'

    os.system(cmd )
