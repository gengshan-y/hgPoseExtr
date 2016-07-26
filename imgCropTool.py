from __future__ import print_function
import sys
import numpy as np
import scipy.misc as imLib
import math

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def crop(img, center, scale, res):
    oriWd, oriHt, oriChan = img.shape
    
    ''' new image range in original img coordinate '''
    tmpSize = math.floor(scale *100)  # convert to int will make it easier to compute size
    center = [math.floor(it) for it in center]
    newUl = [center[0] - tmpSize, center[1] - tmpSize]
    newBr = [center[0] + tmpSize, center[1] + tmpSize]
    
    ''' original image range in new img coordinate '''
    oriUl = np.multiply(newUl, -1)
    oriBr = [oriWd - newBr[0] + tmpSize * 2, oriHt - newBr[1] + tmpSize * 2]

    '''
    print('\nbefore:')
    print('new image in original coordinate:')
    print('upper left: (' + str(newUl[0]) + ', ' + str(newUl[1]) + ')')
    print('bottom right: (' + str(newBr[0]) + ', ' + str(newBr[1]) + ')')
    print('original image in new coordinate:')
    print('upper left: (' + str(oriUl[0]) + ', ' + str(oriUl[1]) + ')')
    print('bottom right: (' + str(oriBr[0]) + ', ' + str(oriBr[1]) + ')')
    '''
    
    ''' generate black new image '''
    newDim = [newBr[0] - newUl[0], newBr[1] - newUl[1], img.shape[2]]
    newImg = np.zeros(newDim)
    
    '''
    print('\nfrom')
    print(img.shape)
    print('to')
    print(newImg.shape)
    '''
    
    ''' crop area of new image beging filled if exceed orignal image range '''
    if oriUl[0] < 0:
        oriUl[0] = 0
        
    if oriUl[1] < 0:
        oriUl[1] = 0
        
    if oriBr[0] > newDim[0]:
        oriBr[0] = newDim[0]
        
    if oriBr[1] > newDim[1]:
        oriBr[1] = newDim[1]
        
    ''' crop area of original image to fill in if exceed new image range  '''
    if newUl[0] < 0:
        newUl[0] = 0
        
    if newUl[1] < 0:
        newUl[1] = 0
        
    if newBr[0] > oriWd:
        newBr[0] = oriWd
        
    if newBr[1] > oriHt:
        newBr[1] = oriHt
     
    '''
    print('\nafter:')
    print('new image in original coordinate:')
    print('upper left: (' + str(newUl[0]) + ', ' + str(newUl[1]) + ')')
    print('bottom right: (' + str(newBr[0]) + ', ' + str(newBr[1]) + ')')
    print('original image in new coordinate:')
    print('upper left: (' + str(oriUl[0]) + ', ' + str(oriUl[1]) + ')')
    print('bottom right: (' + str(oriBr[0]) + ', ' + str(oriBr[1]) + ')')
    '''
    
    newImg[oriUl[0] : oriBr[0], oriUl[1] : oriBr[1]] = img[newUl[0] : newBr[0],
                                                           newUl[1] : newBr[1]]
    newImg = imLib.imresize(newImg, (res, res, 3))
    
    return newImg
