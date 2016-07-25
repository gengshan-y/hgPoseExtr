import numpy as np
import math
import scipy.misc as imLib

def crop(img, center, scale, res):
    oriWd, oriHt, oriChan = img.shape
    
    ''' new image range in original img coordinate '''
    tmpSize = scale *200
    newUl = [math.floor(center[0] - tmpSize/2), math.floor(center[1] - tmpSize/2)]
    newBr = [math.floor(center[0] + tmpSize/2), math.floor(center[1] + tmpSize/2)]
    
    ''' original image range in new img coordinate '''
    oriUl = np.multiply(newUl, -1)
    oriBr = [oriWd - newBr[0] + tmpSize, oriHt - newBr[1] + tmpSize]

    '''
    print('before:')
    print('original coordinate:')
    print('upper left x = ' + str(ul[0]) + ' upper left y = ' + str(ul[1]))
    print('bottom right x = ' + str(br[0]) + ' bottom right y = ' + str(br[1]))
    print('new coordinate:')
    print('upper left x = ' + str(loc_ul[0]) + ' upper left y = ' + str(loc_ul[1]))
    print('bottom right x = ' + str(loc_br[0]) + ' bottom right y = ' + str(loc_br[1]))
    '''
    
    ''' generate black new image '''
    newDim = [newBr[0] - newUl[0], newBr[1] - newUl[1], img.shape[2]]
    newImg = np.zeros(newDim)
    
    ''' crop area of new image beging filled if exceed orignal image range '''
    if oriUl[0] < 0:
        oirUl[0] = 0
        
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
    print('after:')
    print('original coordinate:')
    print('upper left x = ' + str(ul[0]) + ' upper left y = ' + str(ul[1]))
    print('bottom right x = ' + str(br[0]) + ' bottom right y = ' + str(br[1]))
    print('new coordinate:')
    print('upper left x = ' + str(loc_ul[0]) + ' upper left y = ' + str(loc_ul[1]))
    print('bottom right x = ' + str(loc_br[0]) + ' bottom right y = ' + str(loc_br[1]))
    '''
    
    newImg[oriUl[0] : oriBr[0], oriUl[1] : oriBr[1]] = img[newUl[0] : newBr[0],
                                                           newUl[1] : newBr[1]]
    newImg = imLib.imresize(newImg, (res, res, 3))
    
    return newImg
