import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
#input output
def read_pgm(file_name):
    file = open(file_name)
    assert file != None
    lines = file.readlines()
    for l in lines:
        if l[0] == '#':
            lines.remove(l)
    mode = lines[0].split()[0]
    (width,height) = [int(i) for i in lines[1].split()]
    depth = int(lines[2])
    assert depth <= 255
    assert mode == 'P2'
    pixs = []
    for line in lines[3:]:
        for pix in line.split():
            pixs.append(int(pix))
    assert len(pixs) == width * height
    image = np.array(pixs)
    image = image.reshape(height,width)
    file.close()
    return [image, width, height, depth]

def read_ppm(file_name):
    file = open(file_name)
    assert file != None
    lines = file.readlines()
    for l in lines:
        if l[0] == '#':
            lines.remove(l)
    mode = lines[0].split()[0]
    (width,height) = [int(i) for i in lines[1].split()]
    depth = int(lines[2])
    assert depth <= 255
    assert mode == 'P3'
    pixs = []
    for line in lines[3:]:
        for pix in line.split():
            pixs.append(int(pix))
    assert len(pixs) == width * height*3
    r_p = []
    g_p = []
    b_p = []
    for i in range(height*width):
        r_p.append(pixs[3*i])
        g_p.append(pixs[3*i+1])
        b_p.append(pixs[3*i+2])
    image = [r_p,g_p,b_p]
    image = np.array(image)
    image = image.reshape((3,height,width))
    file.close()
    return [image, width, height, depth]
                    
def output_image(image, mode, depth, name):
    o_file = open(name,'w+')
    assert o_file != None
    o_file.write('{} \n'.format(mode))
    if mode == 'P2':
        o_file.write('{} {} \n'.format(image.shape[1],image.shape[0]))
        o_file.write('{} \n'.format(depth))
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                o_file.write('{} '.format(image[r,c]))
            o_file.write('\n')
        o_file.close()
        return
    if mode == 'P3':
        o_file.write('{} {} \n'.format(image.shape[2],image.shape[1]))
        o_file.write('{} \n'.format(depth))
        for r in range(image.shape[1]):
            for c in range(image.shape[2]):
                for k in range(3):
                    o_file.write('{} '.format(image[k,r,c]))
            o_file.write('\n')
        o_file.close()
        return
##method to binarization
# algorithm for adaptive thresholding
# iterativly choosing threshold for binarization
def isoData(image):
    current_t = np.mean(image[:3,:3])
    lower = image[np.where(image <= current_t)]
    upper = image[np.where(image > current_t) ]
    next_t = (np.mean(lower) + np.mean(upper))/2
    while current_t != next_t:
        current_t = next_t
        lower = image[np.where(image <= current_t)]
        upper = image[np.where(image > current_t) ]
        next_t = (np.mean(lower) + np.mean(upper))/2
    return current_t
#convert image to binary format
def to_binary(image,threshold):
    binary = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > threshold:
                binary[i,j] = 1
    return binary.astype(int)

#methmatical morphorlopy
def dilation(image,kernel):
    copy = np.copy(image)
    half_height = int(kernel.shape[0]/2)
    half_width = int(kernel.shape[1]/2)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row,col] == 1:
                for x in range(max(0,row-half_height),min(image.shape[0],row+half_height+1)):
                    for y in range(max(0,col-half_width),min(image.shape[1],col+half_width+1)):
                        copy[x,y] = 1
    return copy

def erosion(image,kernel):
    copy = np.copy(image)
    half_height = int(kernel.shape[0]/2)
    half_width = int(kernel.shape[1]/2)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
                hstart = max(half_height-row,0)
                hend = kernel.shape[0] - max(half_height+row-(image.shape[0]-1),0)
                vstart = max(half_width-col,0)
                vend = kernel.shape[1] - max(half_width+col-(image.shape[1]-1),0)
                eroded = False
                for i in range(hstart,hend):
                    for j in range(vstart,vend):
                        if kernel[i,j] == 1:
                            if image[row-half_height+i, col - half_width + j] != 1:
                                copy[row,col] = 0
                                eroded = True
                        if eroded:
                            break
                    if eroded:
                        break         
    return copy

def closing(image,kernel):
    copy = dilation(image,kernel)
    copy = erosion(copy,kernel)
    return copy

def opening(image,kernel):
    copy = erosion(image, kernel)
    copy = dilation(copy, kernel)
    return copy

# distance map
def contour(binary):
    distance = ndi.distance_transform_edt(binary)
    for row in range(distance.shape[0]):
        for col in range(distance.shape[1]):
            distance[row,col] = distance[row,col]
    return distance

# binary kernel
def disk_kernel(radius):
    kernel = np.ones((radius*2+1,radius*2+1))
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if np.sqrt((i - radius)**2 + (j - radius)**2) > radius:
                kernel[i,j] = 0
    return kernel
def square_kernel(radius):
    kernel = np.ones((radius*2+1,radius*2+1))
    return kernel

#filling small hole in binary image(not used in project but useful)
def hole_filling(image,radius,threshold):
    copy = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = image[max(0,i-radius):min(i+radius,image.shape[0]),max(0,j-radius):min(j+radius,image.shape[1])]
            sum_ = np.sum(window)
            if sum_ < threshold:
                copy[i,j] = 0
    return copy

#routine for finding the minima in the image
def minima_map(image,r):
    minimas = []
    minima_map = np.zeros((3,image.shape[0],image.shape[1]))
    temp_map = np.zeros(image.shape)
    #finding the extrema in the 2*r+1 size square window
    for i in range(image.shape[0]):
        for j in range(image.shape[1]): 
            if image[i,j] != 0 and len(np.where(image[max(i-r,0):min(i+r+1,image.shape[0]),max(0,j-r):min(j+r+1,image.shape[1])] > image[i,j])[0]) == 0:
                minimas.append([i,j])
                temp_map[i,j] = 1 
    #sub routine for marking color of the extrema
    local_close = set()# book keeping set
    color_set = set()# book keeping color set
    for i in range(temp_map.shape[0]):
        for j in range(temp_map.shape[1]):      
            # if not background and not already visted
            if temp_map[i,j] != 0 and not (i,j) in local_close:
                close_list = set()#temp book keeping set holding the element visited
                open_list = set()# a queue to hold the pixel waiting for explored
                open_list.add((i,j))#add it to the queue
                while len(open_list) > 0:
                    current = open_list.pop()
                    close_list.add(current)#set it as visited
                    for k in range(-r,r+1):#in the 2*r+1 neighbor window
                        for l in range(-r,r+1):
                            #if valid
                            if (k== 0 and l == 0) or current[0] + k < 0 or current[0] + k >= temp_map.shape[0] or current[1] + l < 0 or current[1] + l >= temp_map.shape[1]:
                                pass
                            else:
                                #check if in the open list or visted already
                                if temp_map[current[0]+k,current[1]+l] != 0 and not (current[0]+k,current[1]+l) in close_list and not(current[0]+k,current[1]+l) in open_list:
                                    open_list.add((current[0]+k,current[1]+l))
                #random generate color 
                color = (int(np.random.random()*255),int(np.random.random()*255),int(np.random.random()*255))
                while color in color_set and color != 0:
                    color = (int(np.random.random()*255),int(np.random.random()*255),int(np.random.random()*255))
                for pair in close_list:
                    local_close.add(pair)
                    minima_map[0,pair[0],pair[1]] = color[0]
                    minima_map[1,pair[0],pair[1]] = color[1]
                    minima_map[2,pair[0],pair[1]] = color[2]
                color_set.add(color)
    minima_map = minima_map.astype(int)
    return minima_map,len(color_set)# len for color set = number of extrema = number of virus

##routine to segmentation

#Step 1: build a list of queue, the ith queue will contain the pixel with distance i to the background
#Step 2: add every extrema to the queue they correspond to(by the distance of extrema to the background)
#Step 3: pop the queue in highest level(the queue of the pixels that are furthest from the background), find this pixels’ non-background neighborhood pixels, mark them as the same color of the popped pixel, and add them to the corresponding queue(by their distance to background)

def watershed(contour_map,minima_map):
    level = np.max(contour_map)
    level_queue = []
    close = set()
    #step 1
    for i in range(level+1):
        level_queue.append([])
    #step 2
    for i in range(contour_map.shape[0]):
        for j in range(contour_map.shape[1]):
            if not np.array_equal(minima_map[:,i,j],[0,0,0]):
                level_queue[contour_map[i,j]].append((i,j))
                close.add((i,j))
    empty = False
    #step 3
    while not empty:
        empty = True
        select_marker = None
        for i in range(level,0,-1):
            if len(level_queue[i]) != 0:
                select_marker = level_queue[i][0]
                level_queue[i].remove(level_queue[i][0])
                empty = False
                break
        if not empty:
           for k in range(-1,2):
               for l in range(-1,2):
                  if (k== 0 and l == 0) or select_marker[0] + k < 0 or select_marker[0] + k >= contour_map.shape[0] or select_marker[1] + l < 0 or select_marker[1] + l >= contour_map.shape[1]:
                      pass
                  else: 
                      if contour_map[select_marker[0]+k,select_marker[1]+l] != 0:
                          if (select_marker[0]+k,select_marker[1]+l) not in close:
                              close.add((select_marker[0]+k,select_marker[1]+l))
                              print((select_marker[0]+k,select_marker[1]+l))
                              level_queue[contour_map[select_marker[0]+k,select_marker[1]+l]].append((select_marker[0]+k,select_marker[1]+l))
                              minima_map[0,select_marker[0]+k,select_marker[1]+l] = minima_map[0,select_marker[0],select_marker[1]]
                              minima_map[1,select_marker[0]+k,select_marker[1]+l] = minima_map[1,select_marker[0],select_marker[1]]
                              minima_map[2,select_marker[0]+k,select_marker[1]+l] = minima_map[2,select_marker[0],select_marker[1]]
    return 
######## external code not our work
#contrast limit adaptive hitorgram
def clahe(img,clipLimit,nrBins=128,nrX=0,nrY=0):
    '''img - Input image
       clipLimit - Normalized clipLimit. Higher value gives more contrast
       nrBins - Number of graylevel bins for histogram("dynamic range")
       nrX - Number of contextial regions in X direction
       nrY - Number of Contextial regions in Y direction'''
    h,w = img.shape
    if clipLimit==1:
        return
    nrBins = max(nrBins,128)
    if nrX==0:
        #Taking dimensions of each contextial region to be a square of 32X32
        xsz = 32
        ysz = 32
        nrX = int(np.ceil(h/xsz))#240
        #Excess number of pixels to get an integer value of nrX and nrY
        excX= int(xsz*(nrX-h/xsz))
        nrY = int(np.ceil(w/ysz))#320
        excY= int(ysz*(nrY-w/ysz))
        #Pad that number of pixels to the image
        if excX!=0:
            img = np.append(img,np.zeros((excX,img.shape[1])).astype(int),axis=0)
        if excY!=0:
            img = np.append(img,np.zeros((img.shape[0],excY)).astype(int),axis=1)
    else:
        xsz = round(h/nrX)
        ysz = round(w/nrY)
    
    nrPixels = xsz*ysz
    xsz2 = round(xsz/2)
    ysz2 = round(ysz/2)
    claheimg = np.zeros(img.shape)
    
    if clipLimit > 0:
        clipLimit = max(1,clipLimit*xsz*ysz/nrBins)
    else:
        clipLimit = 50
    
    #makeLUT
    print("...Make the LUT...")
    minVal = 0 #np.min(img)
    maxVal = 255 #np.max(img)
    
    #maxVal1 = maxVal + np.maximum(np.array([0]),minVal) - minVal
    #minVal1 = np.maximum(np.array([0]),minVal)
    
    binSz = np.floor(1+(maxVal-minVal)/float(nrBins))
    LUT = np.floor((np.arange(minVal,maxVal+1)-minVal)/float(binSz))
    
    #BACK TO CLAHE
    bins = LUT[img]
    print(bins.shape)
    #makeHistogram
    print("...Making the Histogram...")
    hist = np.zeros((nrX,nrY,nrBins))
    print(nrX,nrY,hist.shape)
    for i in range(nrX):
        for j in range(nrY):
            bin_ = bins[i*xsz:(i+1)*xsz,j*ysz:(j+1)*ysz].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i,j,bin_[i1,j1]]+=1
    
    #clipHistogram
    print("...Clipping the Histogram...")
    if clipLimit>0:
        for i in range(nrX):
            for j in range(nrY):
                nrExcess = 0
                for nr in range(nrBins):
                    excess = hist[i,j,nr] - clipLimit
                    if excess>0:
                        nrExcess += excess
                
                binIncr = nrExcess/nrBins
                upper = clipLimit - binIncr
                for nr in range(nrBins):
                    if hist[i,j,nr] > clipLimit:
                        hist[i,j,nr] = clipLimit
                    else:
                        if hist[i,j,nr]>upper:
                            nrExcess += upper - hist[i,j,nr]
                            hist[i,j,nr] = clipLimit
                        else:
                            nrExcess -= binIncr
                            hist[i,j,nr] += binIncr
                
                if nrExcess > 0:
                    stepSz = max(1,np.floor(1+nrExcess/nrBins))
                    for nr in range(nrBins):
                        nrExcess -= stepSz
                        hist[i,j,nr] += stepSz
                        if nrExcess < 1:
                            break
    
    #mapHistogram
    print("...Mapping the Histogram...")
    map_ = np.zeros((nrX,nrY,nrBins))
    #print(map_.shape)
    scale = (maxVal - minVal)/float(nrPixels)
    for i in range(nrX):
        for j in range(nrY):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i,j,nr]
                map_[i,j,nr] = np.floor(min(minVal+sum_*scale,maxVal))
    
    #BACK TO CLAHE
    #INTERPOLATION
    print("...interpolation...")
    xI = 0
    for i in range(nrX+1):
        if i==0:
            subX = int(xsz/2)
            xU = 0
            xB = 0
        elif i==nrX:
            subX = int(xsz/2)
            xU = nrX-1
            xB = nrX-1
        else:
            subX = xsz
            xU = i-1
            xB = i
        
        yI = 0
        for j in range(nrY+1):
            if j==0:
                subY = int(ysz/2)
                yL = 0
                yR = 0
            elif j==nrY:
                subY = int(ysz/2)
                yL = nrY-1
                yR = nrY-1
            else:
                subY = ysz
                yL = j-1
                yR = j
            UL = map_[xU,yL,:]
            UR = map_[xU,yR,:]
            BL = map_[xB,yL,:]
            BR = map_[xB,yR,:]
            #print("CLAHE vals...")
            subBin = bins[xI:xI+subX,yI:yI+subY]
            #print("clahe subBin shape: ",subBin.shape)
            subImage = interpolate(subBin,UL,UR,BL,BR,subX,subY)
            claheimg[xI:xI+subX,yI:yI+subY] = subImage
            yI += subY
        xI += subX
    
    if excX==0 and excY!=0:
        return claheimg[:,:-excY]
    elif excX!=0 and excY==0:
        return claheimg[:-excX,:]
    elif excX!=0 and excY!=0:
        return claheimg[:-excX,:-excY]
    else:
        return claheimg
def interpolate(subBin,LU,RU,LB,RB,subX,subY):
    subImage = np.zeros(subBin.shape)
    num = subX*subY
    for i in range(subX):
        inverseI = subX-i
        for j in range(subY):
            inverseJ = subY-j
            val = subBin[i,j].astype(int)
            subImage[i,j] = np.floor((inverseI*(inverseJ*LU[val] + j*RU[val])+ i*(inverseJ*LB[val] + j*RB[val]))/float(num))
    return subImage

