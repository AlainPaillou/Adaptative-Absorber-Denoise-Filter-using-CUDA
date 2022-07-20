# coding=utf-8
import time
import numpy as np
import cv2
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.cumath

##################################################################################################
#  Adaptative Absorber Denoise Filter (AADF) - Test program - Copyright Alain Paillou 2018-2022  #
##################################################################################################


# Select the input video for treatment
Video_Test = '/Home/Jetson/Videos/YourVideo.avi' # The path to your video and video name

# Select the output result video
VideoResult = '/Home/Jetson/Videos/Result.avi' # The path to your result video and video name

# Choose quality of the output video
flag_HQ = 0 # if 0 : low quality compressed video - if 1 : high quality RAW video

# Set the dynamic response of the AADF
flag_dyn_AADP = 0 # Choose the filter dynamic - 0 means low dynamic - 1 means high dynamic


# CUDA AADF routine #
mod = SourceModule("""
__global__ void adaptative_absorber_denoise_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *old_r, unsigned char *old_g, unsigned char *old_b,
long int width, long int height, int flag_dyn_AADP)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r,delta_g,delta_b;
  float coef_r,coef_g,coef_b;
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      if (img_r[index] > 235) {
          img_r[index] = (int)((img_r[index-1] + img_r[index+1])/2.0);
          }
      if (img_g[index] > 235) {
          img_g[index] = (int)((img_g[index-1] + img_g[index+1])/2.0);
          }
      if (img_b[index] > 235) {
          img_b[index] = (int)((img_b[index-1] + img_b[index+1])/2.0);
          }
      delta_r = old_r[index] - img_r[index];
      delta_g = old_g[index] - img_g[index];
      delta_b = old_b[index] - img_b[index];
      if (delta_r > 0 && flag_dyn_AADP == 1) {
          coef_r = __powf(abs(delta_r),-0.025995987)*1.2669433195;
      }
      else {
          coef_r = __powf(abs(delta_r),-0.54405)*20.8425; 
      }
      if (delta_g > 0 && flag_dyn_AADP == 1) {
          coef_g = __powf(abs(delta_g),-0.025995987)*1.2669433195;
      }
      else {
          coef_g = __powf(abs(delta_g),-0.54405)*20.8425; 
      }
      if (delta_b > 0 && flag_dyn_AADP == 1) {
          coef_b = __powf(abs(delta_b),-0.025995987)*1.2669433195;
      }
      else {
          coef_b = __powf(abs(delta_b),-0.54405)*20.8425;
      }
      dest_r[index] = (int)((old_r[index] - delta_r / coef_r));
      dest_g[index] = (int)((old_g[index] - delta_g / coef_g));
      dest_b[index] = (int)((old_b[index] - delta_b / coef_b));
      } 
}
""")
adaptative_absorber_denoise_Color = mod.get_function("adaptative_absorber_denoise_Color")


# Initial variables setup
First_frame = True
compteur_FSDN = 0
Im1fsdnOK = False
Im2fsdnOK = False
Flag_Video_Open = True

# Init Video Input and output
videoIn = cv2.VideoCapture(Video_Test)
width = int(videoIn.get(3))
height = int(videoIn.get(4))
if flag_HQ == 0:
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # compressed video
else :
    fourcc = 0 # RAW video
videoOut = cv2.VideoWriter(VideoResult, fourcc, 25, (width, height), isColor = True) # Compressed video

nb_ThreadsX = 16
nb_ThreadsY = 16
nb_blocksX = (width // nb_ThreadsX) + 1
nb_blocksY = (height // nb_ThreadsY) + 1

print("Treatment start")
start_time_test = time.perf_counter()

while Flag_Video_Open == True :
    if (videoIn.isOpened()):
        ret,frame = videoIn.read()
        if ret == True :
            if First_frame == True :
                First_frame = False
                res_b1,res_g1,res_r1 = cv2.split(frame)
                b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                img_b_gpu1 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                img_b_gpu2 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                img_g_gpu1 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                img_g_gpu2 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
            compteur_FSDN = compteur_FSDN + 1
            if compteur_FSDN < 3 :
                
                if compteur_FSDN == 1 :
                    
                    old_frame = frame
                    Im1fsdnOK = True
                if compteur_FSDN == 2 :
                    
                    Im2fsdnOK = True
            res_b1,res_g1,res_r1 = cv2.split(frame)
            if Im2fsdnOK == True :
                nb_images = 2
                res_b1,res_g1,res_r1 = cv2.split(frame)
                res_b2,res_g2,res_r2 = cv2.split(old_frame)
                drv.memcpy_htod(img_b_gpu1, res_b1)  
                drv.memcpy_htod(img_g_gpu1, res_g1)
                drv.memcpy_htod(img_r_gpu1, res_r1)
                drv.memcpy_htod(img_b_gpu2, res_b2)  
                drv.memcpy_htod(img_g_gpu2, res_g2)
                drv.memcpy_htod(img_r_gpu2, res_r2)
                adaptative_absorber_denoise_Color(r_gpu, g_gpu, b_gpu, img_r_gpu1, img_g_gpu1, img_b_gpu1, img_r_gpu2, img_g_gpu2, img_b_gpu2,\
                                     np.int_(width), np.int_(height),np.intc(flag_dyn_AADP),block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                drv.memcpy_dtoh(res_r1, r_gpu)
                drv.memcpy_dtoh(res_g1, g_gpu)
                drv.memcpy_dtoh(res_b1, b_gpu)

                Result_image=cv2.merge((res_b1,res_g1,res_r1))
                old_frame = Result_image
                videoOut.write(Result_image)
        else :
            Flag_Video_Open = False	
            r_gpu.free()
            g_gpu.free()
            b_gpu.free()
            img_r_gpu1.free()
            img_g_gpu1.free()
            img_b_gpu1.free()
            img_r_gpu2.free()
            img_g_gpu2.free()
            img_b_gpu2.free()

videoIn.release()
videoOut.release()

stop_time_test = time.perf_counter()

print("Treatment is OK")

time_exec_test= int((stop_time_test-start_time_test))
print("Treatment time : ",time_exec_test," seconds")
