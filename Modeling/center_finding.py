import itertools
import cv2
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit  # Automatically initializes CUDA, creates a context which is needed


def normalize(im):
    max_mrc=np.max(im)
    min_mrc=np.min(im)
    img_original=(255*((im-min_mrc)/(max_mrc-min_mrc))).astype(np.uint8)
    return(img_original)

def min_rect_circle(cont):
    contours_poly = cv2.approxPolyDP(cont, 3, True)
    center, _= cv2.minEnclosingCircle(contours_poly)
    rect=cv2.minAreaRect(cont)
    box=np.int0(cv2.boxPoints(rect))
    mn,mx=np.amin(box,axis=0),np.amax(box,axis=0)
    diff=mx-mn
    if np.all(diff<(2*radius+40)):
        return(int(center[0]),int(center[1]))
    else:
        pass

def eliminate_near(fields):
    fields=np.array(fields,dtype=np.int32)
    i=np.arange(len(fields))
    diff=fields.reshape(len(fields),1,2)-fields
    D=np.sqrt((diff**2).sum(2))
    D=np.array(D,dtype=np.float64)
    D[np.triu_indices(D.shape[0])]=np.inf
    re = np.where(D< radius)
    out=np.array(list(zip(re[0],re[1])),dtype=np.int32)
    outmin=np.unique(np.min(out,axis=1))
    return(outmin)

def primary_sorting(i):
    #print("primary_sorting done")
    i1,i2,i3,i4=i[1]-(radius*2),i[0]-(radius*2),i[1]+(radius*2),i[0]+(radius*2)
    if i1<0 and i2<0:
        center=(i1+radius*2,i1+radius*2)
        i1=0
        i2=0
    elif i3 > thresh1.shape[0] and i4 > thresh1.shape[1]:
        i3= thresh1.shape[0]
        i4 = thresh1.shape[1]
        center=(radius*2,radius*2)
    elif i1<0:
        center=(i1+radius*2,i1+radius*2)
        i1=0
    elif i2<0:
        center=(i2+radius*2,i2+radius*2)
        i2=0
    elif i3 > thresh1.shape[0]:
        i3= thresh1.shape[0]
        center=(radius*2,radius*2)
    elif i4 > thresh1.shape[1]:
        i4 = thresh1.shape[1]
        center=(radius*2,radius*2)
    else:
        center=(radius*2,radius*2)


    th1=thresh1[i1:i3,i2:i4]

    c_frame=output_image[i1:i3,i2:i4]
    c_frame = cv2.cvtColor(c_frame, cv2.COLOR_GRAY2RGB)
    mask = np.zeros_like(th1)
    mask2 = np.zeros_like(th1)
    mask3 = np.zeros_like(c_frame)
    mask=cv2.circle(mask, center=center, radius=radius,color=(255),thickness=-1)
    mask2=cv2.circle(mask2, center=center, radius=radius+10,color=(255),thickness=-1)
    mask3=cv2.circle(mask3, center=center, radius=radius+10,color=[255,255,255],thickness=-1)
    col_mask= np.bitwise_and(c_frame,mask3)
    result = np.bitwise_and(th1,mask)
    result2 = np.bitwise_and(th1,mask2)
    col_a=np.where(np.all(col_mask== [0,255,0] , axis=-1))
    col_b=np.where(np.all(col_mask== [0,255,255] , axis=-1))
    inn_positions=np.nonzero(result)
    out_positions=np.nonzero(result2)

    contours, hierarchy = cv2.findContours(result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    l=len(contours)
    a,b= (inn_positions[0].max()-inn_positions[0].min()),(inn_positions[1].max()-inn_positions[1].min())
    c,d= (out_positions[0].max()-out_positions[0].min()),(out_positions[1].max()-out_positions[1].min())
    if (len(col_a[0]) or len(col_b[0]))>0 :
        pass
    elif (c or d) > (radius*2.1):
        pass
    elif l==1 and ((c and d) <(radius*2)):
        return(tuple([i[0],i[1]]))
    else:

        #cv2.circle(frame, i, radius, (255, 0, 0), 3)
        ans={}
        centerx=range(0+radius,th1.shape[1]-radius,5)
        centery=range(0+radius,th1.shape[0]-radius,5)
        rad=range(radius,5,5)
        for cx,cy in itertools.product(centerx,centery):
            mask00 = np.zeros_like(th1)
            mask20=mask.copy()
            mask00=cv2.circle(mask00, center=(cx,cy), radius=radius, color=(255,255,255), thickness=-1)
            mask20=cv2.circle(mask20, center=(cx,cy), radius=radius+5, color=(255,255,255), thickness=-1)
            result00 = np.bitwise_and(th1,mask00)
            tot_ellip=np.sum(mask00 == [255])
            tot_ellip_white1=np.sum(result00 == [255])
            per_white=(tot_ellip_white1/tot_ellip)*100
            result002 = np.bitwise_and(th1,mask20)
            tot_ellip_white2=np.sum(result002 == [255])
            diff=tot_ellip_white2-tot_ellip_white1

            if diff<10 and per_white>10:
                ans[per_white]=(cx,cy)


        if len(ans.keys())>0:
            k=ans[max(ans.keys())]
            return (tuple([i2+k[0],i1+k[1]]))
        else:
            pass
            
_mod = SourceModule("""
__global__ void scoreGpu(float* heat,float*gaus, float*score, int sizex, int sizey, int psize)
{
    // 2D Thread ID (assuming that only *one* block will be executed)
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    tx=blockIdx.x*blockDim.x+tx;
    ty=blockIdx.y*blockDim.y+ty;

    int tmp=psize/2;
    if((tx >= tmp) && (ty >= tmp) && (tx <= sizex-tmp) && (ty < sizey-tmp) ){
        int sx=tx-tmp;
        int sy=ty-tmp;
        heat=heat+sx*sizex+sy;
        //heat=heat+tx*sizex+ty;
        float sum=0;
        int sub=0;
        for(int i=0;i<psize;i++){
            for(int j=0;j<psize;j++){
                sum=sum+gaus[sub]*heat[j];
                sub=sub+1;
            }
            heat=heat+sizex;
        }
        score[tx*sizex+ty]=sum;

    }//end if
}
__global__ void getMax(float* score, float* list, int size, int sizex, int sizey, int numx){
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    tx=blockIdx.x*blockDim.x+tx;
    ty=blockIdx.y*blockDim.y+ty;
    int cx=tx*size;
    int cy=ty*size;


    if(cx<sizex&&cy<sizey){
        int sx=size;
        int sy=size;

        if(cx+size>sizex){
            sx=sizex-cx;
        }
        if(cy+size>sizey){
            sy=sizey-cy;
        }
        score=score+cx*sizex+cy;
        //float max=6.805646932770577*(1000000000000000000000);
        float max=0;
        int maxx=0;
        int maxy=0;
        for(int i=0;i<sx;i++){
            for(int j=0;j<sy;j++){
                if(score[j]>=max){
                    max=score[j];
                    maxx=cx+i;
                    maxy=cy+j;
                }
            }
            score=score+sizex;
        }

        int sub=5*(tx*numx+ty);
        list[sub]=maxx;
        list[sub+1]=maxy;
        list[sub+2]=max;
        list[sub+3]=tx;
        list[sub+4]=ty;
    }
}

__global__ void getMax3(float* score, float* list, int psize, int sizex, int sizey, int numx,int num, int iter){
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    tx=blockIdx.x*blockDim.x+tx;
    ty=blockIdx.y*blockDim.y+ty;
    int sub=(tx*numx+ty)*5;

    if (sub<num){
        int cx=list[sub]-psize/2;
        int cy=list[sub+1]-psize/2;

        float max=0;
        int maxx=0;
        int maxy=0;

        for(int i=0;i<iter;i++){

            if(cx<sizex&&cy<sizey&&cx>0&&cy>0){
                int sx=psize;
                int sy=psize;

                if(cx+sx>sizex){
                    sx=sizex-cx;
                }
                if(cy+sy>sizey){
                    sy=sizey-cy;
                }
                max=0;
                maxx=0;
                maxy=0;
                float* score0=score+cx*sizex+cy;
                for(int i=0;i<sx;i++){
                    for(int j=0;j<sy;j++){
                        if(score0[j]>=max){
                            max=score0[j];
                            maxx=cx+i;
                            maxy=cy+j;
                        }
                    }
                    score0=score0+sizex;
                }
                cx=maxx-psize/2;
                cy=maxy-psize/2;
            }
         }//end for

        list[sub]=maxx;
        list[sub+1]=maxy;
        list[sub+2]=max;
        list[sub+3]=tx;
        list[sub+4]=ty;
    }//end if num
}



__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}

""")

# grab Python‐callable wrappers
scoreGpu    = _mod.get_function("scoreGpu")
getMax      = _mod.get_function("getMax")
getMax3     = _mod.get_function("getMax3")
multiply_them = _mod.get_function("multiply_them")

class item:
    def __init__(self,count=0,x=0 ,y=0 ):
        self.id=count
        self.minX = x
        self.minY=y
        self.maxX=x
        self.maxY=y
        self.totalN=0
        self.totalS=0
        self.x=x
        self.y=y
        self.p=0.1
    def __str__(self):
        line=str(self.id)+' '+str(self.minX)+' '+str(self.maxX)+' '+str(self.minY)+' '+str(self.maxY) \
                +' '+str(self.totalN)+' '+str(self.totalS)+' '+str(self.x)+' '+str(self.y)+' '+str(self.p)+'\n'
        return line

    def update(self,x,y):
        if(x<self.minX):
            self.minX=x
        elif(x>self.maxX):
            self.maxX=x

        if(y<self.minY):
            self.minY=y
        elif(y>self.maxY):
            self.maxY=y

        self.totalN=self.totalN+1

        return self

    def getS(self):
        width=self.maxX-self.minX+1
        length=self.maxY-self.minY+1
        self.totalS=width*length
        self.x=self.minX+width/2
        self.y=self.minY+width/2
        self.p=float(self.totalN)/self.totalS
        return self.totalS

def reshape(res,num):
    canList=[]
    for i in range(0,num):
        sub=i*5
        m=res[sub+2]
        if m!=0:
            candi=[int(res[sub]),int(res[sub+1]),m,int(res[sub+2]),int(res[sub+3])]
            canList.append(candi)
    print('convert')
    canList.sort(key=lambda x: x[2], reverse=True)

    #for i in range(0, 10):
    #    print(canList[i])
    return canList

def gaussian_kernel_2d_opencv(kernel_size = 3,sigma = 1):
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    res=np.multiply(kx,np.transpose(ky))
    res=1-(res-np.max(res))/(-1*np.ptp(res))
    return res

def getOverlap(x,y,rx,ry,p):
    t=int(p/2)
    xmin=x-t
    xmax=x+t
    ymin=y-t
    ymax=y+t

    rxmin=rx-t
    rxmax=rx+t
    rymin=ry-t
    rymax=ry+t

    x1=abs(rxmin-xmax)
    x2=abs(rxmax-xmin)
    lx=min(x1,x2)

    y1=abs(rymin-ymax)
    y2=abs(rymax-ymin)
    ly=min(y1,y2)

    return lx*ly

def cleanCanList(canList,op1,psize):
    numCan=len(canList)
    op=op1*psize*psize
    tmp=[-psize,-psize,0]
    for i in range(0,len(canList)):
        candi=canList[i]
        if candi !=tmp:
            x=candi[0]
            y=candi[1]

            for j in range(i+1,len(canList)) :
                candi=canList[j]
                rx=candi[0]
                ry=candi[1]

                if abs(x-rx)<psize and abs(y-ry)<psize:
                    sOp=getOverlap(x,y,rx,ry,psize)
                    if sOp>op:
                        #print(rx,ry,sOp,op)
                        canList[j]=tmp #delete
                        (canList[i])[0]=0.8*x+0.2*rx
                        (canList[i])[1]=0.8*y+0.2*ry


    return [x for x in canList if x != tmp]

def writeStarHead(fstar):
    star=open(fstar,'w')
    star.write('data_\n\n')
    star.write('loop_\n')
    star.write('_rlnCoordinateX #1\n')
    star.write('_rlnCoordinateY #2\n')
    star.write('_rlnParticleSelectZScore  #3\n')
    return star

def writeCan(fstar,canList,sizex, sizey):
    fstar=writeStarHead(fstar)
    p=0
    l=len(canList)
    for i in range(0,l):
        c=canList[i]
        #line=str(c[0])+' '+str(c[1])+' '+str(c[2])+'\n'
        #x=c[1]
        #y=sizey-c[0]
        x=c[0]
        y=c[1]
        x0=y
        y0=x
        if(len(c)>2):
            p=c[2]

        line=str(x0)+' '+str(y0)+' '+str(p)+'\n'
        #print(line)
        fstar.write(line)
    fstar.close()

def reNorm(heatArr, nSep):
    ksize=5

    kernel = np.ones((ksize,ksize),np.uint8)
    img = cv2.erode(heatArr,kernel,iterations = nSep)
    #tmp=int(psize/4)
    #if tmp%2 ==0:
    #    tmp=tmp+1
    #heatArr= cv2.medianBlur(heatArr,tmp)
    #heatArr= cv2.medianBlur(heatArr,5)
    #heatArr= cv2.medianBlur(heatArr,5)
    #heatArr= cv2.medianBlur(heatArr,5)
    return heatArr

def reLev(heatArr, level):
    num=int(255/level)
    heatArr=heatArr/num
    heatArr=heatArr.astype(int)
    return heatArr

def draw(im, cx, cy, sx, sy):
    # Calculate rectangle coordinates
    startx = int(cx - sx / 2)
    starty = int(cy - sy / 2)  # Corrected to use sy
    endx = int(cx + sx / 2)
    endy = int(cy + sy / 2)

    # Calculate colors (simplified for demonstration; adjust as necessary)
    c = 1
    r = (c // 4) * 255
    g = ((c % 4) // 2) * 255
    b = (c % 2) * 255

    # Thickness of the rectangle border
    w = 2

    # Draw the rectangle
    cv2.rectangle(im, (startx, starty), (endx, endy), (b, g, r), w)

    return im

def checkScore(score):
    score0=np.zeros([score.shape[0], score.shape[1],3])
    print(score0.shape, score.shape)
    score0[:,:,0]=score[:,:]
    score0[:,:,1]=score[:,:]
    score0[:,:,2]=score[:,:]
    a=score0
    score0=(255*(a - np.max(a))/-np.ptp(a))
    for c in canList:
        x=int(c[0])
        y=int(c[1])
        sx=sy=psize
        score0=draw(score0,x,y,sx,sy)
        
        
def pad_image(image_array):
    # Get the current height and width of the image
    height, width = image_array.shape

    # Find the smallest pixel value in the image to use as padding value
    min_pixel_value = np.min(image_array)

    # Determine the size of padding needed
    if height > width:
        diff = height - width
        padding = ((0, 0), (0, diff))  # Padding only to the right
    else:
        diff = width - height
        padding = ((0, diff), (0, 0))  # Padding only to the bottom

    # Apply padding, using the smallest pixel value found
    padded_image = np.pad(image_array, padding, mode='constant', constant_values=min_pixel_value)
    return padded_image