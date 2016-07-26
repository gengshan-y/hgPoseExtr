import scipy.misc as imLib
from imgCropTool import *
import string
import os
import sys

batchSize = int(sys.argv[1])
jobID = int(sys.argv[2]) + 1

with open('/home/gengshan/workJul/darknet/results/comp4_det_test_person.txt', 'r') as f:
    wholeDict = f.readlines()

for it, oriDict in enumerate(wholeDict):
    if it < batchSize * (jobID-1):
        continue
    if it >= batchSize * jobID:
        break
    
    oriDict = oriDict.split('\t')
    # print information to stderr
    eprint(str(it) + ' ' + oriDict[1] + ' ' + oriDict[0] + ' ' + 'tmp/test_' \
           + oriDict[0].split('/')[-1])
        
    ''' Get image parameters '''
    img = imLib.imread(oriDict[0])
    wd = float(oriDict[2])
    ht = float(oriDict[3])
    scale = float(oriDict[4])

    ''' Crop image '''
    newImg = crop(img ,[wd, ht], scale, 256)  # width, height
    
    ''' Generate new path '''
    newPath = string.replace(oriDict[0], 'tmp', 'poseTmp')
    if not os.path.isdir(string.replace(newPath, oriDict[0].split('/')[-1], '')):
        os.makedirs(string.replace(newPath, oriDict[0].split('/')[-1], ''))
    
    ''' Save image '''
    imLib.imsave(newPath, newImg)
