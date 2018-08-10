
import cv2
import numpy as np
import queue

class Segmentation:
    threshold=55
    addht=0
    addwd=0
    def __init__(self):
        '''if fn is not None:
            self.xmin,self.ymin=7000,7000
            self.xmax,self.ymax=0,0
            self.pixs=np.array([],dtype=np.uint8)

            self.filename=fn
            self.filename2=fn
        
            self.ax,self.ay=0,0
            self.img=cv2.imread(self.filename,0)
            self.imgOriginal=cv2.imread(self.filename2)
            self.sx,self.sy=self.img.shape'''
    
    def setData(self, fn):
        if fn is not None:
            self.xmin,self.ymin=7000,7000
            self.xmax,self.ymax=0,0
            self.pixs=np.array([],dtype=np.uint8)

            self.filename=fn
            self.filename2=fn
        
            self.ax,self.ay=0,0
            self.imgOriginal=cv2.imread(self.filename2)
            self.img=cv2.cv2.cvtColor(self.imgOriginal, cv2.COLOR_BGR2GRAY)
            if self.img is None:
                print("please pass a valid filename with extension ")
                exit()
            
            self.sx,self.sy=self.img.shape
        
    def resize(self):
        if(self.sx > 3000):
            self.img=Segmentation.image_resize(self.img,height=int(self.sx*.40))
            self.imgOriginal=Segmentation.image_resize(self.imgOriginal,height=int(self.sx*.40))
            self.sx,self.sy=self.img.shape

            '''if(self.sy>1000):
                self.Segmentation.image_resize(self.img,width=1000)
                self.imgOriginal=Segmentation.image_resize(self.imgOriginal,width=1000)
                sx,sy=img.shape'''
        if(self.sx>2000):
            self.img=Segmentation.image_resize(self.img,height=int(self.sx*.50))
            self.imgOriginal=Segmentation.image_resize(self.imgOriginal,height=int(self.sx*.50))
            self.sx,self.sy=self.img.shape
            '''if(sy>1000):
                img=imageResize.image_resize(img,width=1000)
                imgOriginal=imageResize.image_resize(imgOriginal,width=1000)
                sx,sy=img.shape'''

	

    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    def getImage(self):
    
        self.img = cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,\
                cv2.THRESH_BINARY,15,7)
        r,self.img=cv2.threshold(self.img,127,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.sx,self.sy= self.img.shape
        
        r,self.img = cv2.threshold(self.img,127,255,cv2.THRESH_BINARY_INV)
        
        
        kernel = np.ones((1,18),np.uint8)
        
        self.img = cv2.dilate(self.img,kernel,iterations = 1)
        #self.img=cv2.Canny(self.img,100,200)
        #self.img=self.auto_canny(self.img)
        
        self.img=cv2.Laplacian(self.img,cv2.CV_64F)
        #kernel = np.ones((1,18),np.uint8)
        
        #self.img = cv2.dilate(self.img,kernel,iterations = 1)
        
        #kernel = np.ones((2,1),np.uint8)
        #self.img = cv2.erode(self.img,kernel,iterations = 1)
        

        r,self.img = cv2.threshold(self.img,127,255,cv2.THRESH_BINARY_INV)
        
        #self.img=cv2.Canny(self.img,100,200)
        #self.img=cv2.Laplacian(self.img,cv2.CV_32F)
        #self.img=self.auto_canny(self.img)
        #kernel = np.ones((2,15),np.uint8)
        #self.img = cv2.dilate(self.img,kernel,iterations = 1)
        #r,self.img = cv2.threshold(self.img,127,255,cv2.THRESH_BINARY_INV)
        
        cv2.imwrite("blobb_LAP.png",self.img)
    
    @classmethod
    def image_resize(cls, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

            # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    
    

    def segments(self,x,y):
    
        self.xmin,self.ymin=self.img.shape
        self.xmax,self.ymax=0,0
    
        q=queue.Queue()
    
        if(self.img[x][y]==0):
            self.img[x][y]=50
        
            q.put([x,y])
        
            if(self.xmin>x ):
                self.xmin=x
            if(self.ymin>y):
                self.ymin=y
        
            while( 1==1 ):
                #print("1==1")
            
                while( y+1<self.sy and self.img[x][y+1]==0):
                    y=y+1
                    if(self.img[x][y]==0):
                        self.img[x][y]=50
                    
                        q.put([x,y])
            
                if(y>self.ymax):
                    self.ymax=y
                
                while( y+1<self.sy and x+1 < self.sx and self.img[x+1][y+1]==0):
                    y=y+1
                    x=x+1
                    if(self.img[x][y]==0):
                        self.img[x][y]=50
                    
                        q.put([x,y])
            
                if(y>self.ymax):
                    self.ymax=y
                if(x>self.xmax):
                    self.xmax=x
            
                while(y-1>=0 and self.img[x][y-1]==0):
                    y=y-1
                    if(self.img[x][y]==0):
                        self.img[x][y]=50
                    
                        q.put([x,y])
                    
                while(y-1>=0 and x-1>0 and self.img[x-1][y-1]==0):
                    y=y-1
                    x=x-1
                    if(self.img[x][y]==0):
                        self.img[x][y]=50
                    
                        q.put([x,y])
                    
                if(self.ymin>y):
                    self.ymin=y
                if(self.xmin>x):
                    self.xmin=x
            
                while(x-1>=0 and self.img[x-1][y]==0):
                    x=x-1
                    if(self.img[x][y]==0):
                        self.img[x][y]=50
                    
                        q.put([x,y])
                if(self.xmin>x):
                    self.xmin=x
            
                while(x-1>=0 and y+1<self.sy and self.img[x-1][y+1]==0):
                    x=x-1
                    y=y+1
                    if(self.img[x][y]==0):
                        self.img[x][y]=50
                    
                        q.put([x,y])
            
                if(self.xmin>x):
                    self.xmin=x
                if(y>self.ymax):
                    self.ymax=y
            
                while(x+1<self.sx and self.img[x+1][y]==0):
                    x=x+1
                    if(self.img[x][y]==0):
                        self.img[x][y]=50
                    
                        q.put([x,y])
                    
                if(x>self.xmax):
                    self.xmax=x
                
                while(x+1<self.sx  and y-1 > 0 and self.img[x+1][y-1]==0):
                    x=x+1
                    y=y-1
                    if(self.img[x][y]==0):
                        self.img[x][y]=50
                    
                        q.put([x,y])
                if(self.ymin>y):
                    self.ymin=y
                
                if (q.empty()):
                    break
                else:
                    x,y=q.get()

            self.pixs=np.append(self.pixs,[self.xmin,self.ymin,self.xmax,self.ymax])        
                
            
    def fixPix(self):
    
    
        self.pixs=self.pixs.reshape((int(self.pixs.size/4),4))
        
        return

    def findAvgSize(self):
    
        self.ax=np.mean(self.pixs[:,2]-self.pixs[:,0])
        if self.ax+5>Segmentation.threshold :
            Segmentation.threshold=self.ax+5
        
    
    def selectSegments(self):
    
        i=0
    
        while(i<int(self.pixs.size/4)):
        
            if  ( self.pixs[i][2]-self.pixs[i][0] < Segmentation.threshold ):  #ax*3.2
            
                self.pixs= np.delete(self.pixs, i,  axis=0)
                i=i-1
            i=i+1
       
        self.pixs = self.pixs[self.pixs[:,0].argsort()]

        #removing concentric segments if any before
        s=self.pixs.size/4
        j=0
        i=0

        while(i<s):
            j=0
            while(j<i):
                if(self.pixs[j][0] <= self.pixs[i][0] and self.pixs[j][2] >= self.pixs[i][2] ):
                    if(self.pixs[j][1] <= self.pixs[i][1] and self.pixs[j][3] >= self.pixs[i][3]):
                        self.pixs=np.delete(self.pixs,i,axis=0)
                        i-=1
                        s-=1
                        break
                j+=1
            
            i+=1

        '''
        #removing concentric circles if any after
        i,j=0,0
        while(i<s):
            j=i
            while(j<s):
                if(self.pixs[i][0]<self.pixs[j][0] and self.pixs[i][2]>self.pixs[j][2] ):
                    if(self.pixs[i][1]<self.pixs[j][1] and self.pixs[i][3]>self.pixs[j][3]):
                        self.pixs=np.delete(self.pixs,j,axis=0)
                        j-=1
                        s-=1
                        break
                j+=1
            
            i+=1
        '''
        
        return

    def prepareReturn(self):

        i=0
        self.sx,self.sy=self.img.shape
        s=self.pixs.size/4
        while (i<s):
            

	
       
            '''
            #converting pixs into x,y,h,w only
            ht=self.pixs[i][2]-self.pixs[i][0]
            wd=self.pixs[i][3]-self.pixs[i][1]
            self.pixs[i][2]=ht
            self.pixs[i][3]=wd
            '''

            #adding addht to height and addwd to width and converting to x,y,h,w
            ht=self.pixs[i][2]-self.pixs[i][0]
            wd=self.pixs[i][3]-self.pixs[i][1]
            self.pixs[i][0]=self.pixs[i][0]-Segmentation.addht
            self.pixs[i][1]=self.pixs[i][1]-Segmentation.addwd
            self.pixs[i][2]=ht+Segmentation.addht+Segmentation.addht
            self.pixs[i][3]=wd+Segmentation.addwd+Segmentation.addwd

            
	    #code to fix pixs as actual x,y,h,w
            a,b,c,d=self.pixs[i]
            self.pixs[i][0]=b   #x
            self.pixs[i][1]=a   #y
            self.pixs[i][2]=d   #w
            self.pixs[i][3]=c   #h
            

            i+=1

    def makerects(self):
        for x in self.pixs:
            
            #cv2.rectangle(self.imgOriginal,(x[0],x[1]),(x[0]+x[2],x[1]+x[3]),(0,0,255),1)
            cv2.rectangle(self.imgOriginal,(x[0],x[1]),(x[0]+x[2],x[1]+x[3]),(0,0,255),2)
        cv2.imwrite("out.png",self.imgOriginal)

    def doSegmentation(self,filename=None,size_thresh=None):
        if size_thresh is None:
            Segmentation.threshold=55
        #self.__init__(self,filename)
        if filename is None or filename=='' :
            print("please pass filename with extension ")
            print(filename)
            exit()
        
        self.setData(filename)
        self.resize()
        self.getImage()
        i=self.sx
        j=self.sy
        #horizontal checking
        for x in range(0,i-2,4):
            for y in range(0,j-2,4):
                self.segments(x,y)
            
          
        self.fixPix()
        self.findAvgSize()
        
        self.selectSegments()   
            
        self.prepareReturn()
        self.makerects()
        
    
        return self.pixs

if __name__=="__main__":
    obj=Segmentation()
    print (obj.doSegmentation("test4.png",size_thresh=55))

