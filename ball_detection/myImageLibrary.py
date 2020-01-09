import numpy as np
import math
def downsize(image,scale,grayscale = True):
    if scale == 1:
        return np.copy(image)
    if grayscale:
        initX = int(scale/2)
        initY = int(scale/2)
        compact = np.zeros([int(image.shape[0]/scale), int(image.shape[1]/scale)], dtype = int)
        for i in range(compact.shape[0]):
            for j in range(compact.shape[1]):
                compact[i,j] = int(np.mean(image[max(0,initX+(i-1)*scale):min(image.shape[0],initX+i*scale),max(0,initY+(j-1)*scale):min(image.shape[1],initY+j*scale)]))
        return compact    
    else:
        initX = int(scale/2)
        initY = int(scale/2)
        compact = np.zeros([3,int(image.shape[1]/scale), int(image.shape[2]/scale)], dtype = int)
        for i in range(compact.shape[0]):
            for j in range(compact.shape[1]):
                for k in range(compact.shape[2]):
                    compact[i,j,k] = int(np.mean(image[i,max(0,initX+(j-1)*scale):min(image.shape[1],initX+j*scale),max(0,initY+(k-1)*scale):min(image.shape[2],initY+k*scale)]))#image[i,j*scale,k*scale]
        return compact            
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
    
def guassian_density(x,sigma):
    return 1/(math.sqrt(2*math.pi)*sigma)*math.exp(-x**2/(2*sigma**2))

def makeGaussian1d():
    conv = []
    sigma = 1.5
    for i in range(-int(2*sigma-1),int(2*sigma)):##i choose to do 2 steps smooth
        conv.append(guassian_density(i,sigma))
    for i in range(len(conv)):
        conv[i] = 256*conv[i]
    s = sum(conv)
    for i in range(len(conv)):
        conv[i] = conv[i]/s ##normalize it
    return np.array(conv)

def makeGaussian2d():
    conv = []
    sigma = 1.5
    for i in range(-int(2*sigma-1),int(2*sigma)):
        for j in range(-int(2*sigma-1),int(2*sigma)):
            conv.append(guassian_density(i,sigma)*guassian_density(j,sigma))       
    for i in range(len(conv)):
        conv[i] = 256*conv[i]
    s = sum(conv)
    for i in range(len(conv)):
        conv[i] = conv[i]/s 
    size = int(math.sqrt(len(conv)))
    return np.array(conv).reshape(size,size)

def convolve2d(image,kernel, renorm = True):
    assert kernel.shape[0]%2 == 1
    half_size = int(kernel.shape[0]/2)
    result = np.zeros(image.shape, dtype = int)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            hstart = max(half_size-r,0)
            hend = kernel.shape[0] - max(half_size+r-(image.shape[0]-1),0)
            vstart = max(half_size-c,0)
            vend = kernel.shape[1] - max(half_size+c-(image.shape[1]-1),0)
            s= np.sum(image[r-half_size+hstart:r-half_size+hend, c-half_size+vstart:c-half_size+vend]*kernel[hstart:hend,vstart:vend])
            if renorm:
                s = s/np.sum(kernel[hstart:hend,vstart:vend])
            result[r,c] = round(s)
    return result

def convolveH1d(image,kernel,renorm = True):
    assert kernel.shape[0]%2 == 1
    half_size = int(kernel.shape[0]/2)
    result = np.zeros(image.shape, dtype = int)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            hstart = max(half_size-c,0)
            hend = kernel.shape[0] - max(half_size+c-(image.shape[1]-1),0)
            s = np.sum(image[r,c-half_size+hstart:c-half_size+hend]*kernel[hstart:hend])
            if renorm:
                s = s/np.sum(kernel[hstart:hend])
            result[r,c] = round(s)
    return result

def convolveV1d(image,kernel,renorm = True):
    assert kernel.shape[0]%2 == 1
    half_size = int(kernel.shape[0]/2)
    result = np.zeros(image.shape, dtype = int)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            hstart = max(half_size-r,0)
            hend = kernel.shape[0] - max(half_size+r-(image.shape[0]-1),0)
            s = np.sum(image[r-half_size+hstart:r-half_size+hend,c]*kernel[hstart:hend])
            if renorm:
                s = s/np.sum(kernel[hstart:hend])            
            result[r,c] = round(s)
    return result

def convolvePartition2D(image,kernel, renorm = True):
    result = convolveH1d(image,kernel,renorm = renorm)
    result = convolveV1d(result,kernel,renorm = renorm)
    return result

def image_gradient(image):
    assert image.shape[0] > 1 and image.shape[1] > 1
    gradients_m = []
    gradients_d = []
    kernel = [-0.5,0,0.5]
    for row in range(image.shape[0]):
        gradient_m_in_row = []
        gradient_d_in_row = []
        for col in range(image.shape[1]):
            x_d = 0
            y_d = 0
            if row == 0:
                x_d = image[row+1,col] - image[row,col]
            elif row == image.shape[0]-1:
                x_d = image[row,col] - image[row-1,col]
            else:
                x_d = np.sum(image[row-1:row+2,col] * kernel)
            if col == 0:
                y_d = image[row,col+1] - image[row,col]
            elif col == image.shape[1]-1:
                y_d = image[row,col] - image[row,col-1]
            else:
                y_d = np.sum(image[row,col-1:col+2] * kernel)
            theta = np.arctan(y_d/x_d)
            if np.isnan(theta):
                theta = np.pi
            gradient = int(np.sqrt(x_d*x_d + y_d*y_d))
            gradient_m_in_row.append(gradient)
            gradient_d_in_row.append(theta)
        gradients_m.append(gradient_m_in_row)
        gradients_d.append(gradient_d_in_row)
    gradients_m = np.array(gradients_m)
    gradients_d = np.array(gradients_d)
    return gradients_m,gradients_d

def edge_thinning(g_m,g_d):
    h_interval = [-np.pi/8, np.pi/8]#interval for horizontal gradient
    p_interval = [np.pi/8, np.pi*3/8]#interval for +1 gradient
    n_interval = [-np.pi*3/8, -np.pi/8]#interval for -1 gradient
    copy = np.copy(g_m)
    for row in range(g_m.shape[0]):
        for col in range(g_m.shape[1]):
            if g_d[row,col] < h_interval[1] and g_d[row,col] >= h_interval[0]:#gradient is in the horizontal direction
                if row > 0 and row < g_m.shape[0]-1: ## if it is not in boundry
                    if g_m[row,col] < g_m[row-1,col] or g_m[row,col] < g_m[row+1,col]:
                        copy[row,col] = 0
                elif row == 0:##boundry
                    if g_m[row,col] < g_m[row+1,col]:
                        copy[row,col] = 0
                else:#boundry
                    if g_m[row,col] < g_m[row-1,col]:
                        copy[row,col] = 0 
            elif g_d[row,col] < p_interval[1] and g_d[row,col] >= p_interval[0]:#gradient in the +1 diretion
                if row > 0 and row < g_m.shape[0]-1 and col > 0 and col < g_m.shape[1]-1:
                    if g_m[row,col] < g_m[row+1,col+1] or g_m[row,col] < g_m[row-1,col-1]:
                        copy[row,col] = 0
                elif (row == 0 and col == g_m.shape[1]-1) or (row == g_m.shape[0]-1 and col == 0):##boundry
                    pass
                elif row == 0 or col == 0:##boundry
                    if g_m[row,col] < g_m[row+1,col+1]:
                        copy[row,col] = 0
                else:##boundry
                    if g_m[row,col] < g_m[row-1,col-1]:
                        copy[row,col] = 0
            elif g_d[row,col] < n_interval[1] and g_d[row,col] >= n_interval[0]:#gradient in the -1 direction
                if row > 0 and row < g_m.shape[0]-1 and col > 0 and col < g_m.shape[1]-1:
                    if g_m[row,col] < g_m[row+1,col-1] or g_m[row,col] < g_m[row-1,col+1]:
                        copy[row,col] = 0
                elif (row == 0 and col == 0) or (row == g_m.shape[0]-1 and col == g_m.shape[1]-1):##boundry
                    pass
                elif row == 0 or col == g_m.shape[1]-1:##boundry
                    if g_m[row,col] < g_m[row+1,col-1]:
                        copy[row,col] = 0
                else:##boundry
                    if g_m[row,col] < g_m[row-1,col+1]:
                        copy[row,col] = 0 
            else:## gradient in vertical direction
                if col > 0 and col < g_m.shape[1]-1:
                    if g_m[row,col] < g_m[row,col-1] or g_m[row,col] < g_m[row,col+1]:
                        copy[row,col] = 0
                elif col == 0:##boundry
                    if g_m[row,col] < g_m[row,col+1]:
                        copy[row,col] = 0
                else:##boundry
                    if g_m[row,col] < g_m[row,col-1]:
                        copy[row,col] = 0 
    return copy
            
def noise_suppress(g_m):
    low_threshold = 0.15*np.max(g_m)
    high_threshold = 0.2*np.max(g_m)
    for row in range(g_m.shape[0]):
        for col in range(g_m.shape[1]):
            if g_m[row,col] < low_threshold:
                g_m[row,col] = 0
            elif g_m[row,col] < high_threshold:
                if np.max(g_m[max(0,row-1):min(row+1,g_m.shape[0]-1),max(0,col-1):min(col+1,g_m.shape[1]-1)]) < high_threshold:
                    g_m[row,col] = 0
    return g_m

def canny_edge_detection(image,smooth = True, grayscale = True):
    if grayscale:
        g_d = None
        g_m = None
        if smooth:
            g_m, g_d = image_gradient(convolvePartition2D(image,makeGaussian1d()))
        else:
            g_m, g_d = image_gradient(image)
        edge_map = noise_suppress(edge_thinning(g_m, g_d))
        return edge_map.astype(int)
    else:# rgb version of canny edge detection
        g_d = []
        g_m = []
        for i in range(3):
            smooth = convolvePartition2D(image[i],makeGaussian1d())
            m, d = image_gradient(smooth)
            g_d.append(d)
            g_m.append(m)
        real_g_m = np.zeros([image.shape[1],image.shape[2]])
        real_g_d = np.zeros([image.shape[1],image.shape[2]])
        #get the maximum gradient in three channel as the gradient in edge map
        for row in range(image.shape[1]):
            for col in range(image.shape[2]):
                max_channel = -1
                max_val = 0
                for i in range(3):
                    if g_m[i][row,col] > max_val:
                        max_channel = i
                        max_val = g_m[i][row,col]
                real_g_m[row,col] = max_val
                real_g_d[row,col] = g_d[max_channel][row,col]
        edge_map = noise_suppress(edge_thinning(real_g_m, real_g_d))
        return edge_map.astype(int)
    
#hough transformer
class Hough_Transformer:
    ## input: downscale the portion will be used to downscale the image 
    def __init__(self,downscale = 1):
        self.downscale = downscale
    ## input: image_name: the name of the image contain circle need to be detect
    #         output_name: the name of the highlighted image
    #         grayscale: whether the image is grayscale
    #         supportthreshold: determine what threshold of the voting circle will get highlighted
    #         estimate radius: an interval with lowerbound and upperbound of the estimate ration of circle radius to the longer side of the image
    #         highlight: the color of the highlight
    #         the width of the highlight
    def detect_circle(self,image_name, output_name, grayscale = True,support_threshold = 1, estimate_ratio = [0.01,0.5], highlight = 255, highlight_width = 1,need_smooth = True):
        if grayscale:# if the imgae is grayscale, read it in pgm
            [image, width, height, depth] = read_pgm(image_name)
        else:# if it is rgb, read it in PPM
            [image, width, height, depth] = read_ppm(image_name)
        #downscale the image by the downscale
        compact = downsize(image,self.downscale, grayscale = grayscale)
        if grayscale:#output the downscale result
            output_image(compact,'P2',255,'downscale_image.pgm')
        else:
            output_image(compact,'P3',255,'downscale_image.ppm')
        #use canny edge detection to detect the edge in the image
        edge_map = canny_edge_detection(compact,smooth = need_smooth, grayscale = grayscale)
        #visualization
        output_image(edge_map,'P2',255,'edge_map.pgm')
        #find the index of nonzero pixel in the edge map
        X,Y = np.nonzero(edge_map)
        #number of nonzero pixel
        not_zero_num = X.shape[0]
        #set the maximum and minimum radius by the specify radius
        maxradius = int(max(compact.shape[0],compact.shape[1])*estimate_ratio[1])
        minradius = int(max(compact.shape[0],compact.shape[1])*estimate_ratio[0])
        #initialize the voying space
        tensor_3d = np.zeros([maxradius,edge_map.shape[0],edge_map.shape[1]])
        #for every nonzero number
        for i in range(not_zero_num):
            #visualization of voting process
            print('checking pixels {}/{}'.format(i,not_zero_num))
            # from min radius to max_radius, voting
            for j in range(minradius,maxradius):
                #calculate the location of the edge of each circle
                x_y_bound= self.calculate_circle(X[i],Y[i],j+1,edge_map.shape[0],edge_map.shape[1])
                for k in range(x_y_bound.shape[0]):#voting in there location
                    tensor_3d[j,x_y_bound[k][0],x_y_bound[k][1]] = tensor_3d[j,x_y_bound[k][0],x_y_bound[k][1]] + 1    
        center = []#list for holding center
        max_crossing_point = np.unravel_index(np.argmax(tensor_3d, axis=None),tensor_3d.shape)#the location of maxvote
        max_support = tensor_3d[max_crossing_point]#the vote
        tensor_3d[max_crossing_point] = 0#set it to 0
        threshold = max_support * support_threshold#initialize the voting threshold
        while max_support >= threshold:#loop to find the center that meet the threshold
            max_crossing_point = list(max_crossing_point)
            max_crossing_point[0] = max_crossing_point[0]+1#adjust the radius by one since the index 1 pixel less than real radius
            center.append(max_crossing_point)#add to the center list
            max_crossing_point = np.unravel_index(np.argmax(tensor_3d, axis=None),tensor_3d.shape)
            max_support  = tensor_3d[max_crossing_point]
            tensor_3d[max_crossing_point] = 0
        #delete the redundant circle
        if grayscale:
            self.delete_repeat_center(center,compact.shape)
        else:
            self.delete_repeat_center(center,[compact.shape[1],compact.shape[2]])
        #invert the circle from downscale space to original space
        self.inverse_transform_center(center,self.downscale)
        #output the highlighted image
        if output_name != None:  
            if grayscale:
                self.highlight_circle(center,image,highlight,highlight_width,grayscale = grayscale)
                output_image(image, 'P2', depth, output_name)
            else:
                self.highlight_circle(center,image,highlight,highlight_width,grayscale = grayscale)
                output_image(image, 'P3', depth, output_name)
        return center
    #highlight the circle in the image   
    def highlight_circle(self,center,image,hightlight, highlight_width, grayscale = True):
        if grayscale:
            for i in range(len(center)):
                #calculate the edge of the circle and then draw it on the image
                x_y_path = self.calculate_circle(center[i][1],center[i][2],center[i][0],image.shape[0],image.shape[1])
                for j in range(x_y_path.shape[0]):
                    image[x_y_path[j][0]-highlight_width:x_y_path[j][0]+highlight_width+1,x_y_path[j][1]-highlight_width:x_y_path[j][1]+highlight_width+1] = hightlight
            return image
        else:
            for i in range(len(center)):
                x_y_path = self.calculate_circle(center[i][1],center[i][2],center[i][0],image.shape[1],image.shape[2])
                for j in range(x_y_path.shape[0]):
                    for c in range(3):
                        image[c,x_y_path[j][0]-highlight_width:x_y_path[j][0]+highlight_width+1,x_y_path[j][1]-highlight_width:x_y_path[j][1]+highlight_width+1] = hightlight
            return image
    #calculate the edge of the circle
    def calculate_circle(self,x,y,r,maxX,maxY):
        x_y_boundary = []
        upper_bound = r*8#this due to the fact that the length of the edge is less than the length of the square enclose this circle
        theta = 2*np.pi/(upper_bound)
        for i in range(upper_bound):
            newx = int(x + r*np.cos(theta*i))
            newy = int(y + r*np.sin(theta*i))
            if [newx,newy] in x_y_boundary:
                pass
            elif newx < 0 or newy < 0 or newx > maxX-1 or newy > maxY-1:
                pass
            else:
                x_y_boundary.append([newx,newy])
        return (np.array(x_y_boundary))
    #inverse transform the circle to the original space there might be a bit rounding error
    def inverse_transform_center(self,center,scale):
        for i in range(len(center)):
            center[i][0] = center[i][0] * scale
            center[i][1] = int((center[i][1]-1) * scale + scale/2)
            center[i][2] = int((center[i][2]-1) * scale + scale/2)
    # the similarity of two vector
    def similarity(self,x,y):
        distance = 0
        for i in range(len(x)):
            distance = distance + (x[i] - y[i])**2
        return 1 - distance
    #delete the repeat circle
    #firstly using minmax sclae to normalize
    def delete_repeat_center(self,center,size):
        copy = np.copy(center).tolist()
        for i in range(len(copy)):
            copy[i][0] = float(copy[i][0]/max(size[0],size[1]))
            copy[i][1] = float(copy[i][1]/size[0])
            copy[i][2] = float(copy[i][2]/size[1]) 
        i = 0
        while i < len(center)-1:
            j = i + 1
            while j < len(center):
                sim = self.similarity(copy[i],copy[j])
                if sim > 0.99:
                    center.remove(center[j])
                    copy.remove(copy[j])
                else:
                    j = j + 1
            i = i + 1
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        