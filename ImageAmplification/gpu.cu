/**************************************************************************************
 *The gpu.cu performs are the cannyEdge detection,Vertical and Horizontal Edge keeping
 *and mean keeping to obtain a High resolution image from low resolution image input
 *Various kernel functions are invoked to perform the above mentioned operations.
**************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include"cuda_runtime_api.h"
#include<math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "gpu.h"


__constant__ float *d_gx_mask;
__constant__ float *d_gy_mask;
__constant__ float *mask;


__shared__ float Nshared[BLOCKWIDTH+2*(g_w/2)][BLOCKWIDTH+2*(g_w/2)];
__device__
void prepNshared(float *image,int width,int height){

                int i,j;
                //NORTH
                if(blockIdx.x==0)
                {
                        if(threadIdx.x >= BLOCKWIDTH - g_w/2)
                        {
                                Nshared[threadIdx.x+g_w/2 - BLOCKWIDTH][threadIdx.y+g_w/2]=0;
                        }
                }
                else
                {
                        if(threadIdx.x >=BLOCKWIDTH - g_w/2)
                        {
                                i=threadIdx.x+(blockIdx.x-1)*blockDim.x;
                                j=threadIdx.y+blockIdx.y*blockDim.y;
                                Nshared[threadIdx.x+g_w/2 - BLOCKWIDTH][threadIdx.y+g_w/2]=
                                image[i*width+j];
                        }
                }

		//for south elements

                if(blockIdx.x==gridDim.x - 1)
                {
                        if(threadIdx.x < g_w/2)
                        {
                                 Nshared[threadIdx.x + BLOCKWIDTH + g_w/2][threadIdx.y+g_w/2]=0; 
                        }
                }
                else
                {
                        if(threadIdx.x < g_w/2)
                        {
                                i= threadIdx.x +(blockIdx.x+1)*blockDim.x;
                                j=threadIdx.y+(blockIdx.y *blockDim.y);
                                Nshared[threadIdx.x + BLOCKWIDTH + g_w/2][threadIdx.y+g_w/2]=image[i*width+j];
                        }
                }
                //for west elements

                if(blockIdx.y==0)
                {
                        if(threadIdx.y>= BLOCKWIDTH-g_w/2)
                        {
                                Nshared[threadIdx.x +g_w/2][threadIdx.y+g_w/2 - BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if(threadIdx.y>= BLOCKWIDTH-g_w/2)
                        {
                                i= threadIdx.x +blockIdx.x *blockDim.x;
                                j = threadIdx.y+(blockIdx.y-1)*blockDim.y;
                                Nshared[threadIdx.x +g_w/2][threadIdx.y+g_w/2 - BLOCKWIDTH]=image[i*width+j];
                        }
                }


                //for east elements

                if(blockIdx.y==gridDim.y-1)
                {
                        if(threadIdx.y < g_w/2) //VKP:Initially BLOCKWIDTH-g_w/2
                        {
                                Nshared[threadIdx.x +g_w/2][threadIdx.y+g_w/2+BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if(threadIdx.y < g_w/2) //VKP:Initially BLOCKWIDTH-g_w/2
                        {
                                i=threadIdx.x+blockIdx.x*blockDim.x;
                                j = threadIdx.y +(blockIdx.y+1)*blockDim.y;
                                Nshared[threadIdx.x +g_w/2][threadIdx.y+g_w/2+BLOCKWIDTH]=image[i*width+j];
                        }
                }

                //for north west elements

                if(blockIdx.x==0 || blockIdx.y==0)
                {
                        if((threadIdx.x >= BLOCKWIDTH-g_w/2)&&(threadIdx.y >= BLOCKWIDTH - g_w/2))
                        {
                                Nshared[threadIdx.x+g_w/2-BLOCKWIDTH][threadIdx.y+g_w/2 - BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if((threadIdx.x >= BLOCKWIDTH-g_w/2)&&(threadIdx.y >= BLOCKWIDTH - g_w/2))
                        {
                                i= threadIdx.x+(blockIdx.x -1)*blockDim.x;
                                j = threadIdx.y +(blockIdx.y -1)*blockDim.y;
                                Nshared[threadIdx.x+g_w/2-BLOCKWIDTH][threadIdx.y+g_w/2 - BLOCKWIDTH]=image[i*width+j];
                        }
                }
                //for north east elements

                if((blockIdx.x==0) || (blockIdx.y == gridDim.y-1))
                {
                        if((threadIdx.x >= BLOCKWIDTH - g_w/2)&&(threadIdx.y <g_w/2))
                        {
                                Nshared[threadIdx.x +g_w/2-BLOCKWIDTH][threadIdx.y+g_w/2+BLOCKWIDTH]=0; 
                        }
                }
                else
                {
                        if((threadIdx.x >= BLOCKWIDTH - g_w/2)&&(threadIdx.y <g_w/2))
                        {
                                i= threadIdx.x +(blockIdx.x -1)*blockDim.x;
                                j = threadIdx.y +(blockIdx.y +1)*blockDim.y;
                                Nshared[threadIdx.x +g_w/2 - BLOCKWIDTH][threadIdx.y+g_w/2+BLOCKWIDTH]=image[i*width+j] =image[i*width+j]; 
                        }
                }

                //for south west elements

                if((blockIdx.x==gridDim.x-1)||(blockIdx.y==0))
                {
                        if((threadIdx.x<g_w/2)&&(threadIdx.y >=BLOCKWIDTH- g_w/2)) 
                        {
                                Nshared[threadIdx.x+g_w/2+BLOCKWIDTH][threadIdx.y+g_w/2 - BLOCKWIDTH] =0; 
                        }
                }
                else
                {
                        if((threadIdx.x <g_w/2)&&(threadIdx.y>= BLOCKWIDTH-g_w/2))
                        {
                                i= threadIdx.x +(blockIdx.x +1)*blockDim.x;
                                j = threadIdx.y +(blockIdx.y -1)*blockDim.y;
                                Nshared[threadIdx.x+g_w/2+BLOCKWIDTH][threadIdx.y+g_w/2 - BLOCKWIDTH] =image[i*width+j];
                        }
                }

                //forsouth east elements

                if((blockIdx.x == gridDim.x -1) || (blockIdx.y == gridDim.y-1))
                {
                        if((threadIdx.x <g_w/2)&&(threadIdx.y <g_w/2))
                        {
                                Nshared[threadIdx.x +g_w/2+BLOCKWIDTH][threadIdx.y+g_w/2+BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if((threadIdx.x <g_w/2)&&(threadIdx.y <g_w/2))
                        {
                                i=threadIdx.x +(blockIdx.x+1)*blockDim.x;
                                j = threadIdx.y+(blockIdx.y+1)*blockDim.y;
                        Nshared[threadIdx.x +g_w/2+BLOCKWIDTH][threadIdx.y+g_w/2+BLOCKWIDTH]=image[i*width+j];
                        }
                }

                i= threadIdx.x+blockIdx.x*blockDim.x;
                j=threadIdx.y+blockIdx.y*blockDim.y;

                Nshared[threadIdx.x +(g_w/2)][threadIdx.y+(g_w/2)]=image[i*width+j];

                __syncthreads ();

}

__device__
int gpu_compare(int v,int d, int ad)
{
	int a=0; //default vertical

	if(v>d)
	{
		if(v>ad)
			a=0;
		else
			a=-1;

	}
	else
	{
		if(d>ad)
			a=1;
		else
			a=-1;
	}
	return a;
}


//kernel for magAngle and suppression
__global__
void gpu_conv_mag_phase(float *image,int width,int height,float *d_Gxy_outimage,float *d_Igx_mask,float *d_Igy_mask,float *d_I_angle,float *d_suppressed){
        int i,j;
        float temp=0;
        float temp1=0;
        int tid1 = threadIdx.x+blockIdx.x*blockDim.x;
        int tid2= threadIdx.y+blockIdx.y*blockDim.y;
        if((threadIdx.x+blockIdx.x*blockDim.x)<height && (threadIdx.y + blockIdx.y*blockDim.y)<width)
        {
          prepNshared(image,width,height);

         for(i=0;i<g_w;i++)
          {
                for(j=0;j<g_w;j++)
                {
                        temp = temp+d_Igx_mask[i*g_w+j]*Nshared[i+threadIdx.x][j+threadIdx.y];
                        temp1 = temp1+d_Igy_mask[i*g_w+j]*Nshared[i+threadIdx.x][j+threadIdx.y];
                }
        }
        d_Gxy_outimage[tid1*width+tid2]=sqrt(temp*temp+temp1*temp1);
        d_suppressed[tid1*width+tid2]=sqrt(temp*temp+temp1*temp1);
        d_I_angle[tid1*width+tid2] = atan2(temp, temp1);
 	} 
}


__global__
void gpu_doubleThreshold(float *d_hyst,float *d_suppressed,int width,int height,float th_high,float th_low){

int tid1 = threadIdx.x+blockIdx.x*blockDim.x;
int tid2 = threadIdx.y+blockIdx.y*blockDim.y;

	if(tid1<height && tid2<width)
	{
		if (d_suppressed[tid1*width+tid2]>th_high)
		{
			d_hyst[tid1*width+tid2]=(float)255;
			d_suppressed[tid1*width+tid2]=(float)255;
		}
		else if (d_suppressed[tid1*width+tid2]<th_high && d_suppressed[tid1*width+tid2]>th_low)
		{
			d_hyst[tid1*width+tid2]=(float)125;
			d_suppressed[tid1*width+tid2]=(float)125;
		}
		else 
		{
			d_hyst[tid1*width+tid2]=0;
			d_suppressed[tid1*width+tid2]=0;
		}
	}
}

//edge linking

__global__
void gpu_edgeLinking(float *d_buffer,float *d_hyst,int width,int height){

int tid1 = threadIdx.x+blockIdx.x*blockDim.x;
int tid2 = threadIdx.y+blockIdx.y*blockDim.y;

        if(tid1<height && tid2<width)
	{	
		if(d_buffer[tid1*width+tid2]==(float)125)
		{	
			d_hyst[tid1*width+tid2]=(float)0;
			if(tid1-1>=0)
			{
                           	if(d_buffer[(tid1-1)*width +tid2]==(float)255)
				{
                                      d_hyst[tid1*width+tid2]=(float)255;
				}
			}
                        if(tid1+1<height)
			{
                              	if(d_buffer[(tid1+1)*width + tid2]==(float)255)
			      	{
                                   d_hyst[tid1*width+tid2]=(float)255;
				}	
			}
				
			//left and right (i,j-1) and (i,j+1)
			if(tid1-1>=0)
			{
                        	if(d_buffer[tid1*width + (tid2-1)]==(float)255)
				{
                                	d_hyst[tid1*width + tid2]=(float)255;
				}
			}
                        if(tid2+1<width)
			{
                        	if(d_buffer[tid1*width + (tid2+1)]==255)
				{
                                	d_hyst[tid1*width + tid2]=(float)255;
				}
			}
				
			//Diagonal Pixels (tid1-1,tid2-1) and (tid1+1,tid2+1)
			if((tid1-1) >=0 && (tid2-1) >=0)
			{
                        	if(d_buffer[tid1*width + (tid2-1)]==(float)255)
				{
                                	d_hyst[tid1*width + tid2]=(float)255;
				}
			}
                        if(tid1+1<height && (tid2+1) <width)
			{
                        	if(d_buffer[(tid1+1)*width + (tid2+1)]==(float)255)
				{
                                	d_hyst[tid1*width + tid2]=(float)255;
				}
			}
			
			//Anti-Diagonal Pixels (tid1-1,tid2+1) (tid1+1,tid2-1)
			if((tid1-1)>=0 && (tid2+1) <width)
			{
                        	if(d_buffer[(tid1-1)*width + (tid2+1)]==(float)255)
				{
                                	d_hyst[tid1*width + tid2]=(float)255;
				}
			}
                        if((tid1+1)<height && (tid2-1) >=0)
			{
                        	if(d_buffer[(tid1+1)*width + (tid2-1)]==(float)255)
				{
                                	d_hyst[tid1*width + tid2]=(float)255;
				}
			}
                  }
	}
}


//device shared kernel for nonmaximal suppression
__shared__ float Nshared1[BLOCKWIDTH+2*(apron/2)][BLOCKWIDTH+2*(apron/2)];
__device__
void prepNshared1(float *image,int width,int height){

                int i,j;
                //NORTH
                if(blockIdx.x==0)
                {
                        if(threadIdx.x >= BLOCKWIDTH - apron/2)
                        {
                                Nshared1[threadIdx.x+apron/2 - BLOCKWIDTH][threadIdx.y+apron/2]=0;
                        }
                }
                else
                {
                        if(threadIdx.x >=BLOCKWIDTH - apron/2)
                        {
                                i=threadIdx.x+(blockIdx.x-1)*blockDim.x;
                                j=threadIdx.y+blockIdx.y*blockDim.y;
                                Nshared1[threadIdx.x+apron/2 - BLOCKWIDTH][threadIdx.y+apron/2]=
                                image[i*width+j];
                        }
                }

                //for south elements

                if(blockIdx.x==gridDim.x - 1)
                {
                        if(threadIdx.x < apron/2)
                        {
                                 Nshared1[threadIdx.x + BLOCKWIDTH + apron/2][threadIdx.y+apron/2]=0; 
                        }
                }
                else
                {
                        if(threadIdx.x < apron/2)
                        {
                                i= threadIdx.x +(blockIdx.x+1)*blockDim.x;
                                j=threadIdx.y+(blockIdx.y *blockDim.y);
                                Nshared1[threadIdx.x + BLOCKWIDTH + apron/2][threadIdx.y+apron/2]=image[i*width+j];//VKP:Initially theadIdx.x + BLOCKWIDTH - apron/2
                        }
                }
                //for west elements
if(blockIdx.y==0)
                {
                        if(threadIdx.y>= BLOCKWIDTH-apron/2)
                        {
                                Nshared1[threadIdx.x +apron/2][threadIdx.y+apron/2 - BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if(threadIdx.y>= BLOCKWIDTH-apron/2)
                        {
                                i= threadIdx.x +blockIdx.x *blockDim.x;
                                j = threadIdx.y+(blockIdx.y-1)*blockDim.y;
                                Nshared1[threadIdx.x +apron/2][threadIdx.y+apron/2 - BLOCKWIDTH]=image[i*width+j];
                        }
                }


                //for east elements

                if(blockIdx.y==gridDim.y-1)
                {
                        if(threadIdx.y < apron/2) //VKP:Initially BLOCKWIDTH-apron/2
                        {
                                Nshared1[threadIdx.x +apron/2][threadIdx.y+apron/2+BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if(threadIdx.y < apron/2) //VKP:Initially BLOCKWIDTH-apron/2
                        {
                                i=threadIdx.x+blockIdx.x*blockDim.x;
                                j = threadIdx.y +(blockIdx.y+1)*blockDim.y;
                                Nshared1[threadIdx.x +apron/2][threadIdx.y+apron/2+BLOCKWIDTH]=image[i*width+j];
                        }
                }

                //for north west elements

                if(blockIdx.x==0 || blockIdx.y==0)
                {
                        if((threadIdx.x >= BLOCKWIDTH-apron/2)&&(threadIdx.y >= BLOCKWIDTH - apron/2))
                        {
                                Nshared1[threadIdx.x+apron/2-BLOCKWIDTH][threadIdx.y+apron/2 - BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if((threadIdx.x >= BLOCKWIDTH-apron/2)&&(threadIdx.y >= BLOCKWIDTH - apron/2))
                        {
                                i= threadIdx.x+(blockIdx.x -1)*blockDim.x;
                                j = threadIdx.y +(blockIdx.y -1)*blockDim.y;
                                Nshared1[threadIdx.x+apron/2-BLOCKWIDTH][threadIdx.y+apron/2 - BLOCKWIDTH]=image[i*width+j];
                        }
                }
//for north east elements

                if((blockIdx.x==0) || (blockIdx.y == gridDim.y-1))
                {
                        if((threadIdx.x >= BLOCKWIDTH - apron/2)&&(threadIdx.y <apron/2))
                        {
                                Nshared1[threadIdx.x +apron/2-BLOCKWIDTH][threadIdx.y+apron/2+BLOCKWIDTH]=0; 
                        }
                }
                else
                {
                        if((threadIdx.x >= BLOCKWIDTH - apron/2)&&(threadIdx.y <apron/2))
                        {
                                i= threadIdx.x +(blockIdx.x -1)*blockDim.x;
                                j = threadIdx.y +(blockIdx.y +1)*blockDim.y;
                                Nshared1[threadIdx.x +apron/2 - BLOCKWIDTH][threadIdx.y+apron/2+BLOCKWIDTH]=image[i*width+j] =image[i*width+j]; 
                        }
                }

                //for south west elements

                if((blockIdx.x==gridDim.x-1)||(blockIdx.y==0))
                {
                        if((threadIdx.x<apron/2)&&(threadIdx.y >=BLOCKWIDTH- apron/2)) 
                        {
                                Nshared1[threadIdx.x+apron/2+BLOCKWIDTH][threadIdx.y+apron/2 - BLOCKWIDTH] =0; 
                        }
                }
                else
                {
                        if((threadIdx.x <apron/2)&&(threadIdx.y>= BLOCKWIDTH-apron/2))
                        {
                                i= threadIdx.x +(blockIdx.x +1)*blockDim.x;
                                j = threadIdx.y +(blockIdx.y -1)*blockDim.y;
                                Nshared1[threadIdx.x+apron/2+BLOCKWIDTH][threadIdx.y+apron/2 - BLOCKWIDTH] =image[i*width+j]; 
                        }
                }

                //forsouth east elements

                if((blockIdx.x == gridDim.x -1) || (blockIdx.y == gridDim.y-1))
                {
                        if((threadIdx.x <apron/2)&&(threadIdx.y <apron/2))
                        {
                                Nshared1[threadIdx.x +apron/2+BLOCKWIDTH][threadIdx.y+apron/2+BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if((threadIdx.x <apron/2)&&(threadIdx.y <apron/2))
                        {
                                i=threadIdx.x +(blockIdx.x+1)*blockDim.x;
                                j = threadIdx.y+(blockIdx.y+1)*blockDim.y;
                        Nshared1[threadIdx.x +apron/2+BLOCKWIDTH][threadIdx.y+apron/2+BLOCKWIDTH]=image[i*width+j];
                        }
                }

                i= threadIdx.x+blockIdx.x*blockDim.x;
                j=threadIdx.y+blockIdx.y*blockDim.y;

                Nshared1[threadIdx.x +(apron/2)][threadIdx.y+(apron/2)]=image[i*width+j];
 		__syncthreads ();

}             

//non maximal suppression

__global__
void gpu_suppress(float *d_Gxy_outimage,float *d_suppressed,float *d_I_angle,int width,int height){
	float theta=0;
        float mag=0;
        int tid1 = threadIdx.x+blockIdx.x*blockDim.x;
        int tid2= threadIdx.y+blockIdx.y*blockDim.y;
        if((threadIdx.x+blockIdx.x*blockDim.x)<height && (threadIdx.y + blockIdx.y*blockDim.y)<width)
        {
          prepNshared1(d_Gxy_outimage,width,height);

         theta=(180/M_PI)*d_I_angle[tid1*width+tid2];
                        if(theta<0)
                                theta+=(float)180;
                        mag=Nshared1[threadIdx.x+(apron/2)][threadIdx.y+(apron/2)];
                        if(theta>(157.5) || theta <=22.5) //Left and Right
                        {
                                if((tid2-1) >=0)
				{
                                  if(mag<Nshared1[threadIdx.x+(apron/2)][threadIdx.y+(apron/2) -1])
                        	  {         
			             d_suppressed[tid1*width+tid2]=(float)0;
				  }
				}
                                if((tid2+1)<width)
				{
                                        if(mag<Nshared1[threadIdx.x+(apron/2)][threadIdx.y+(apron/2)+1])
					{
                                                d_suppressed[tid1*width+tid2]=(float)0;
                        		}
				}

			}

                        if(theta>(22.5) && theta <=67.5) //Diagonal pixels
                        {
                                if((tid1-1)>=0 && (tid2-1) >=0)
				{
                                        if(mag<Nshared1[threadIdx.x+(apron/2)-1][threadIdx.y+(apron/2)-1])
                                	{ 
				              d_suppressed[tid1*width+tid2]=(float)0;
					}
				}
                                if((tid1+1)<height && (tid2+1) <width)
				{
                                        if(mag<Nshared1[threadIdx.x+(apron/2)+1][threadIdx.y+(apron/2)+1])
					{
                                                d_suppressed[tid1*width+tid2]=(float)0;
					}
				}
                        }

                        if(theta>(67.5) && theta <= 112.5) //top and bottom. 
                        {
                                if((tid1-1)>=0)
				{
                                        if(mag<Nshared1[threadIdx.x+(apron/2)-1][threadIdx.y+(apron/2)])
                        		{          
			              		d_suppressed[tid1*width+tid2]=(float)0;
					}
				}
                                if((tid1+1)<height)
				{
                                        if(mag<Nshared1[threadIdx.x+(apron/2)+1][threadIdx.y+(apron/2)])
                                	{ 
				               d_suppressed[tid1*width+tid2]=(float)0;
					}
				}
                        }
                        if(theta>(112.5) && theta <= 157.5) //Anti Diagonal.
                        {
                                if((tid1+1)<height && (tid2-1)>=0)
				{
                                        if(mag<Nshared1[threadIdx.x+(apron/2)+1][threadIdx.y+(apron/2)-1])
					{
                                                d_suppressed[tid1*width+tid2]=(float)0;
					}
				}
                                if((tid1-1) >=0 &&(tid2+1)<=width)
				{
                                        if(mag<Nshared1[threadIdx.x+(apron/2)-1][threadIdx.y+(apron/2)+1])
					{
                                                d_suppressed[tid1*width+tid2]=(float)0;
					}
				}


                        }
        }
	
}

__global__
void gpu_verticalEdgeKeeping(float *d_hyst,float *d_I_angle,float *d_image,int h_im_width,int h_im_height,int h_alpha,float *d_HRV,float *d_map){

int tid = threadIdx.y+blockIdx.y*blockDim.y;
int vertical,diagonal,anti_diagonal;
int vert_start=0;
float theta=90;
int direction=0;
int count=0;
int ND=0;
int k =0,m=0,n;
int start_x =0,start_x_amp=0,start_y_amp=0;
int end_x =0,end_x_amp=0,end_y_amp=0;
int rodsize=0;

//Enclose everything inside if(tid <h_im_width)

if(tid <h_im_width) {
	while(vert_start<h_im_height) //not to exceed the height of the image
	{
		while(d_hyst[vert_start*h_im_width + tid]==0 && vert_start<h_im_height) //Hunting for the start of the candidate rod
		{
			vert_start++;
		}
		start_x=vert_start; //Found the head of the vertical rod
	
		while(d_hyst[vert_start*h_im_width + tid]!=0 && vert_start < h_im_height) //Traverse that candidate rod and find its end
		{
			//to get the start point of the rod
			end_x=vert_start;
			vert_start++;
		}

		//Rod traversed. Head of the vertical rod is start_x and tail is end_x
	
		rodsize = end_x - start_x + 1;	
	
		//Check the orientation
	
		vertical=0;diagonal=0,anti_diagonal=0;
	
		for(k=start_x;k<=end_x;k++)
		{
			//Fill the edge map. Initially each candidate rod IS a rod
				
			d_map[k*h_im_width + tid]=(float)255;

			theta=d_I_angle[k*h_im_width + tid];
				
			theta= (180/PI)*theta;
		
			if(theta<0)
				theta+=(float)180;

			if(theta > 22.5 && theta <=67.5) //diagonal +1
				diagonal++;
			else if(theta>112.5 && theta<=157.5) //anti-diagonal -1
				anti_diagonal++;
			else if(theta>67.5 && theta<=112.5) //vertical 0
				vertical++;

		} 

		direction= gpu_compare(vertical,diagonal,anti_diagonal);
		
		//Orientation check finished
	
		ND=0;
	
		//Qualify or disqualify the candidate rod

		if(vertical==0 && diagonal==0 && anti_diagonal==0)
		{
			//Unkown direction for Vertical rod. Disqualifying the candidate rod by setting ND=1
			ND=1;
		}
	
		//If the candidate rod is disqualified, note in the map using weak pixel (125)
		if(ND==1)
		{
			for(k=start_x;k<=end_x;k++)
			{
				d_map[k*h_im_width+tid]=(float)125;
			}
		}	
	
		//Replicate the qualified rods		
		if(direction==0 && ND==0) //replicate vertically
		{
			//These are for amplified image
			start_x_amp=h_alpha*start_x; 
			start_y_amp=h_alpha*tid;
				
			end_x_amp=h_alpha*end_x;
			end_y_amp=h_alpha*tid;
				
			count=0;

			for(k=0;k<h_alpha;k++) //no. of times (alpha) to replicate
			{		
				for(m=0;m<rodsize;m++)
				{			
					if((start_x_amp+k*rodsize+m)<h_alpha*h_im_height && start_y_amp<h_alpha*h_im_width)
						d_HRV[((start_x_amp+k*rodsize+m)*h_alpha*h_im_width) + start_y_amp]=(float)d_image[(start_x+m)*h_im_width + (tid)];
					
					//Bottom pixels need to be filled
					for(n=1;n<=(h_alpha-1);n++)
						if((start_x_amp+k*rodsize+m)<h_alpha*h_im_height && (start_y_amp+n)<h_alpha*h_im_width && (tid+1)<h_im_width)
							d_HRV[(start_x_amp+k*rodsize+m)*h_alpha*h_im_width + start_y_amp+n]=(float)d_image[(start_x+(int)(count/h_alpha))*h_im_width + (tid+1)];
					count++;		
				}
			}
		}
		
		else if(direction==1 && ND==0) //diagonal
		{

			start_x_amp=h_alpha*start_x; 
			start_y_amp=h_alpha*tid + direction*(h_alpha -1);
			end_x_amp=h_alpha*end_x;
			end_y_amp=h_alpha*tid;

			count=0;
			for(k=0;k<h_alpha;k++) //Number of times to repeat this rod
			{
				for(m=0;m<rodsize;m++)
				{
					if((start_x_amp + k*rodsize+m)<h_alpha*h_im_height &&(start_y_amp-1*k*direction)<h_alpha*h_im_width&&(start_y_amp-1*k*direction)>=0)
					{
			 			d_HRV[(start_x_amp + k*rodsize+m)*h_alpha*h_im_width + (start_y_amp-1*k*direction)]=(float)d_image[(start_x+m)*h_im_width + tid];
					}

					//if k=0, fill pixels to the left
					if(k==0)
					{
						for(n=1;n<=(h_alpha-1);n++)
						{                                        
                					if((start_x_amp + k*rodsize+m)<h_alpha*h_im_height &&(start_y_amp-1*k*direction-n)>=0 && (start_y_amp-1*k*direction-n)<h_alpha*h_im_width && ((tid)-1)>=0) //copy adjacent pixel 

                                                               d_HRV[(start_x_amp + k*rodsize+m)*h_alpha*h_im_width + start_y_amp-1*k*direction-n]=(float)d_image[(start_x+(int)(count/h_alpha))*h_im_width + (tid -1)];
							 
                                                	count++;
						}	
					}

					//if k=1, fill pixels to the right
					else
					{
						for(n=1;n<=(h_alpha-1);n++)
						{
                                                        if((start_x_amp + k*rodsize+m)<h_alpha*h_im_height &&(start_y_amp-1*k*direction+n)<h_alpha*h_im_width && (start_y_amp-1*k*direction+n)>=0 && (tid+1)<h_im_width) //copy adjacent pixel
                                                                	d_HRV[(start_x_amp + k*rodsize+m)*h_alpha*h_im_width + start_y_amp-1*k*direction+n]=(float)d_image[(start_x+(int)(count/h_alpha))*h_im_width + (tid+1)];
                                                	count++;
						}
					}
				}
			}
		}//else if ends here

		else if(direction==-1 && ND==0)//anti-diagonal
		{

			start_x_amp=h_alpha*start_x; 
			start_y_amp=h_alpha*tid; //the rod direction is perpendicular to the gradient. Starts at same y position as vertical
			end_x_amp=h_alpha*end_x;
			end_y_amp=h_alpha*tid;

			count=0;
			for(k=0;k<h_alpha;k++) //Number of times to repeat this rod
			{
				for(m=0;m<rodsize;m++)
				{	
					if((start_x_amp + k*rodsize+m)<h_alpha*h_im_height &&(start_y_amp-1*k*direction)<h_alpha*h_im_width&&(start_y_amp-1*k*direction)>=0)
						d_HRV[(start_x_amp + k*rodsize+m)*h_alpha*h_im_width + (start_y_amp-1*k*direction)]=(float)d_image[(start_x+m)*h_im_width + (tid)];

					if(k==0) //fill to the right
					{
						for(n=1;n<=(h_alpha-1);n++)
                                                       	if((start_x_amp + k*rodsize+m)<h_alpha*h_im_height &&(start_y_amp-1*k*direction+n)<h_alpha*h_im_width && (start_y_amp-1*k*direction+n)>=0 && (tid+1)<h_im_width) //copy adjacent pixel
                                                               	d_HRV[(start_x_amp + k*rodsize+m)*h_alpha*h_im_width + start_y_amp-1*k*direction+n]=(float)d_image[(start_x+(int)(count/h_alpha))*h_im_width + (tid+1)];
                                                       count++;
					}
					else //fill to the left
					{
						for(n=1;n<=(h_alpha-1);n++)
                                                	if((start_x_amp + k*rodsize+m)<h_alpha*h_im_height &&(start_y_amp-1*k*direction-n)>=0 && (start_y_amp-1*k*direction-n)<h_alpha*h_im_width && (tid-1)>=0) //copy adjacent pixel
                                                                       d_HRV[(start_x_amp + k*rodsize+m)*h_alpha*h_im_width + start_y_amp-1*k*direction-n]=(float)d_image[(start_x+(int)(count/h_alpha))*h_im_width + (tid-1)];
                                                  count++;
					}
				}
			}
		}
		 

	} //Outermost while loop

} //End the if statement on all threads
		
} //function ends here


__global__
void gpu_horizontalEdgeKeeping(float *d_hyst,float *d_I_angle,float *d_image,int h_im_width,int h_im_height,int h_alpha,float *d_HRH,float *d_maph){


int tid = threadIdx.x+blockIdx.x*blockDim.x;
int k,m,n;
int horizontal,diagonal,anti_diagonal;
int horizontal_start=0;
float theta=0;
int direction=0;
int start_x=0,start_y=0,end_x=0,end_y=0;
int start_x_amp=0,start_y_amp=0; //Start and stop positions for filling in the HR Image
int count=0;
float ND=0;
int rodsize=0;

if(tid <h_im_height){
	while(horizontal_start<h_im_width) //not to exceed the width of the image
		{
			//go to the start of the horizontal rod
			while(d_hyst[tid*h_im_width+horizontal_start]==0 && horizontal_start<h_im_width)
			{
				horizontal_start++;
			}
			start_y= horizontal_start;

			while(d_hyst[tid*h_im_width + horizontal_start]!=0 && horizontal_start<h_im_width)
			{
				end_y = horizontal_start;
				horizontal_start++;
			}
			
			rodsize = end_y - start_y +1;


		//Check the orientation
	
			horizontal=0;diagonal=0;anti_diagonal=0;
			
			for(k=start_y;k<=end_y;k++)
			{
				//Each rod IS a rod
				d_maph[tid*h_im_width+k]=(float)255;
				theta=d_I_angle[tid*h_im_width+k];
				theta = (180/PI)*theta;
				if(theta<0)
					theta+=(float)180;

				if(theta>22.5 && theta<=67.5) //diagonal
					diagonal++;
				else if(theta>112.5 && theta<=157.5) //anti-diagonal
					anti_diagonal++;
				else if(theta>(157.5) || theta <=22.5) //horizontal
					horizontal++;
			}
			direction= gpu_compare(horizontal,diagonal,anti_diagonal);
			
			ND = 0;

			if(horizontal==0 && diagonal==0 && anti_diagonal==0)
			{
				ND=1;
			}
			if(ND==1)
			{
				for(k=start_y;k<=end_y;k++)
				{
					d_maph[tid*h_im_width+k]=(float)125;
				}
			}
			if(direction==0 && ND==0) //replicate horizontally
			{
				start_x_amp=h_alpha*tid;
				start_y_amp=h_alpha*start_y;
				
				end_x=h_alpha*tid;
				end_y=h_alpha*end_y;
		
				count=0;

				for(k=0;k<h_alpha;k++) //no. of times to replicate
				{
					
					for(m=0;m<rodsize;m++)
					{			
						if(start_x_amp<h_alpha*h_im_height && (start_y_amp+k*rodsize+m)<h_alpha*h_im_width)
						//HR[(start_x*alpha*width) + (start_y+k*rod_y.size()+m)]=(float)255;	
							d_HRH[(start_x_amp*h_alpha*h_im_width) + (start_y_amp+k*rodsize+m)]=(float)d_image[(tid)*h_im_width + (start_y+m)];
						//Bottom pixels need to be filled
						for(n=1;n<=(h_alpha-1);n++)
							if((start_x_amp+n)<h_alpha*h_im_height && (start_y_amp+k*rodsize+m)<h_im_width && (tid+1)<h_im_height)
								d_HRH[(start_x_amp+n)*h_alpha*h_im_width + (start_y_amp+k*rodsize+m)]=(float)d_image[(tid+1)*h_im_width + (start_y+(int)(count/h_alpha))];

						count++;		
					}
				}
			}
			
			else if(direction==1 && ND==0) //diagonal direction, replicate upwards
			{

				start_x=h_alpha*tid + direction*(h_alpha-1);
				start_y=h_alpha*start_y;				

				count=0;
				for(k=0;k<h_alpha;k++)
				{
					for(m=0;m<rodsize;m++)
					{
						if((start_x_amp -1*k*direction)<h_alpha*h_im_height && (start_x_amp -1*k*direction)>=0 && (start_y_amp + k*rodsize+m)<h_alpha*h_im_width && (start_y_amp + k*rodsize+m)>=0)
							d_HRH[(start_x_amp -1*k*direction)*h_alpha*h_im_width + (start_y_amp + k*rodsize+m)]=(float)d_image[(tid)*h_im_width + start_y+m];
						//Top pixels need to be filled if k==0
						if(k==0)
							for(n=1;n<=(h_alpha-1);n++)
								if((start_x_amp-1*k*direction-n)>=0 && (start_x_amp-1*k*direction-n)<h_alpha*h_im_height && (start_y_amp+k*rodsize+m)<h_im_width && (tid-1)>=0)
									d_HRH[(start_x_amp-1*k*direction-n)*h_alpha*h_im_width + (start_y_amp+k*rodsize+m)]=(float)d_image[(tid-1)*h_im_width + (start_y+(int)(count/h_alpha))];
								
						else //fill the bottom pixels
							for(n=1;n<=(h_alpha-1);n++)        
								if((start_x_amp-1*k*direction+n)<h_alpha*h_im_height && (start_x_amp-1*k*direction+n)>=0 && (start_y_amp+k*rodsize+m)<h_im_width && (tid+1)<h_im_height)

                                                                        d_HRH[(start_x_amp-1*k*direction+n)*h_alpha*h_im_width + (start_y_amp+k*rodsize+m)]=(float)d_image[(tid+1)*h_im_width + (start_y+(int)(count/h_alpha))];	
	
						count++;
					}
				}

			

			}
		
			else if(direction==-1 &&ND==0)//anti-diagonal direction, replicate downwards 
			{
				start_x=h_alpha*tid;
				start_y=h_alpha*start_y;
				count=0;


				for(k=0;k<h_alpha;k++)
				{

					for(m=0;m<rodsize;m++)
					{
						if((start_x_amp -1*k*direction)<h_alpha*h_im_height && (start_x_amp -1*k*direction)>=0 && (start_y_amp + k*rodsize+m)<h_alpha*h_im_width && (start_y_amp + k*rodsize+m)>=0)
		
							d_HRH[(start_x_amp-1*k*direction)*h_alpha*h_im_width + (start_y_amp + k*rodsize + m)]=(float)d_image[(tid)*h_im_width + start_y];
				
						if(k==0) //fill bottom pixels
							for(n=1;n<=(h_alpha-1);n++)                                                             
								if((start_x_amp-1*k*direction+n)<h_alpha*h_im_height && (start_x_amp-1*k*direction+n)>=0 && (start_y_amp+k*rodsize+m)<h_im_width && (tid+1)<h_im_height)
                                                	        	d_HRH[(start_x_amp-1*k*direction+n)*h_alpha*h_im_width + (start_y_amp+k*rodsize+m)]=(float)d_image[(tid+1)*h_im_width + (start_y+(int)(count/h_alpha))];


						else //fille the top
							for(n=1;n<=(h_alpha-1);n++)                                                             
								if((start_x_amp-1*k*direction-n)>=0 && (start_x_amp-1*k*direction-n)<h_alpha*h_im_width && (start_y_amp+k*rodsize+m)<h_im_width && (tid-1)>=0)
                                                                        d_HRH[(start_x_amp-1*k*direction-n)*h_alpha*h_im_width + (start_y_amp+k*rodsize+m)]=(float)d_image[(tid-1)*h_im_width + (start_y+(int)(count/h_alpha))];
						count++;
					}
				}		

			}
		}
	}
}



//device shared for d_hyst
__shared__ float Nshared2[BLOCKWIDTH+2*(apron/2)][BLOCKWIDTH+2*(apron/2)];
__device__
void prepNshared2(float *image,int width,int height){

                int i,j;
                //NORTH
                if(blockIdx.x==0)
                {
                        if(threadIdx.x >= BLOCKWIDTH - apron/2)
                        {
                                Nshared2[threadIdx.x+apron/2 - BLOCKWIDTH][threadIdx.y+apron/2]=0;
                        }
                }
                else
                {
                        if(threadIdx.x >=BLOCKWIDTH - apron/2)
                        {
                                i=threadIdx.x+(blockIdx.x-1)*blockDim.x;
                                j=threadIdx.y+blockIdx.y*blockDim.y;
                                Nshared2[threadIdx.x+apron/2 - BLOCKWIDTH][threadIdx.y+apron/2]=
                                image[i*width+j];
                        }
                }

                //for south elements

                if(blockIdx.x==gridDim.x - 1)
                {
                        if(threadIdx.x < apron/2)
                        {
                                 Nshared2[threadIdx.x + BLOCKWIDTH + apron/2][threadIdx.y+apron/2]=0; //VKP:Initially theadIdx.x + BLOCKWIDTH - apron/2
                        }
                }
                else
                {
                        if(threadIdx.x < apron/2)
                        {
                                i= threadIdx.x +(blockIdx.x+1)*blockDim.x;
                                j=threadIdx.y+(blockIdx.y *blockDim.y);
                                Nshared2[threadIdx.x + BLOCKWIDTH + apron/2][threadIdx.y+apron/2]=image[i*width+j];//VKP:Initially theadIdx.x + BLOCKWIDTH - apron/2
                        }
                }
                //for west elements
if(blockIdx.y==0)
                {
                        if(threadIdx.y>= BLOCKWIDTH-apron/2)
                        {
                                Nshared2[threadIdx.x +apron/2][threadIdx.y+apron/2 - BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if(threadIdx.y>= BLOCKWIDTH-apron/2)
                        {
                                i= threadIdx.x +blockIdx.x *blockDim.x;
                                j = threadIdx.y+(blockIdx.y-1)*blockDim.y;
                                Nshared2[threadIdx.x +apron/2][threadIdx.y+apron/2 - BLOCKWIDTH]=image[i*width+j];
                        }
                }


                //for east elements

                if(blockIdx.y==gridDim.y-1)
                {
                        if(threadIdx.y < apron/2) //VKP:Initially BLOCKWIDTH-apron/2
                        {
                                Nshared2[threadIdx.x +apron/2][threadIdx.y+apron/2+BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if(threadIdx.y < apron/2) //VKP:Initially BLOCKWIDTH-apron/2
                        {
                                i=threadIdx.x+blockIdx.x*blockDim.x;
                                j = threadIdx.y +(blockIdx.y+1)*blockDim.y;
                                Nshared2[threadIdx.x +apron/2][threadIdx.y+apron/2+BLOCKWIDTH]=image[i*width+j];
                        }
                }

                //for north west elements

                if(blockIdx.x==0 || blockIdx.y==0)
                {
                        if((threadIdx.x >= BLOCKWIDTH-apron/2)&&(threadIdx.y >= BLOCKWIDTH - apron/2))
                        {
                                Nshared2[threadIdx.x+apron/2-BLOCKWIDTH][threadIdx.y+apron/2 - BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if((threadIdx.x >= BLOCKWIDTH-apron/2)&&(threadIdx.y >= BLOCKWIDTH - apron/2))
                        {
                                i= threadIdx.x+(blockIdx.x -1)*blockDim.x;
                                j = threadIdx.y +(blockIdx.y -1)*blockDim.y;
                                Nshared2[threadIdx.x+apron/2-BLOCKWIDTH][threadIdx.y+apron/2 - BLOCKWIDTH]=image[i*width+j];
                        }
                }
//for north east elements

                if((blockIdx.x==0) || (blockIdx.y == gridDim.y-1))
                {
                        if((threadIdx.x >= BLOCKWIDTH - apron/2)&&(threadIdx.y <apron/2))
                        {
                                Nshared2[threadIdx.x +apron/2-BLOCKWIDTH][threadIdx.y+apron/2+BLOCKWIDTH]=0; //VKP:mistake corrected in Nshared2[][*]
                        }
                }
                else
                {
                        if((threadIdx.x >= BLOCKWIDTH - apron/2)&&(threadIdx.y <apron/2))
                        {
                                i= threadIdx.x +(blockIdx.x -1)*blockDim.x;
                                j = threadIdx.y +(blockIdx.y +1)*blockDim.y;
                                Nshared2[threadIdx.x +apron/2 - BLOCKWIDTH][threadIdx.y+apron/2+BLOCKWIDTH]=image[i*width+j] =image[i*width+j]; //VKP: mistake corrected in Nshared2[][*]
                        }
                }

                //for south west elements

                if((blockIdx.x==gridDim.x-1)||(blockIdx.y==0))
                {
                        if((threadIdx.x<apron/2)&&(threadIdx.y >=BLOCKWIDTH- apron/2)) //VKP:there was a mistake with threadIdx.y condition. Check test.cu for mistake
                        {
                                Nshared2[threadIdx.x+apron/2+BLOCKWIDTH][threadIdx.y+apron/2 - BLOCKWIDTH] =0; //VKP: you forgot to apron/2
                        }
                }
                else
                {
                        if((threadIdx.x <apron/2)&&(threadIdx.y>= BLOCKWIDTH-apron/2))//VKP:there was a mistake with threadIdx.y condition. Check test.cu for mistake
                        {
                                i= threadIdx.x +(blockIdx.x +1)*blockDim.x;
                                j = threadIdx.y +(blockIdx.y -1)*blockDim.y;
                                Nshared2[threadIdx.x+apron/2+BLOCKWIDTH][threadIdx.y+apron/2 - BLOCKWIDTH] =image[i*width+j]; //you forgot to apron/2
                        }
                }

                //forsouth east elements

                if((blockIdx.x == gridDim.x -1) || (blockIdx.y == gridDim.y-1))
                {
                        if((threadIdx.x <apron/2)&&(threadIdx.y <apron/2))
                        {
                                Nshared2[threadIdx.x +apron/2+BLOCKWIDTH][threadIdx.y+apron/2+BLOCKWIDTH]=0;
                        }
                }
                else
                {
                        if((threadIdx.x <apron/2)&&(threadIdx.y <apron/2))
                        {
                                i=threadIdx.x +(blockIdx.x+1)*blockDim.x;
                                j = threadIdx.y+(blockIdx.y+1)*blockDim.y;
                        Nshared2[threadIdx.x +apron/2+BLOCKWIDTH][threadIdx.y+apron/2+BLOCKWIDTH]=image[i*width+j];
                        }
                }

                i= threadIdx.x+blockIdx.x*blockDim.x;
                j=threadIdx.y+blockIdx.y*blockDim.y;

                Nshared2[threadIdx.x +(apron/2)][threadIdx.y+(apron/2)]=image[i*width+j];
 		__syncthreads ();

}             


__global__
void gpu_meanKeeping(float *d_image,float *d_hyst,int h_im_width,int h_im_height,int h_alpha,float *d_map,float *d_maph,float *d_high_res){

float patch3_00=0;
float patch3_01=0;
float patch3_02=0;
float patch3_10=0;
float patch3_11=0;
float patch3_12=0;
float patch3_20=0;
float patch3_21=0;
float patch3_22=0;
float patcha_00=0;
float patcha_01=0;
float patcha_10=0;
float patcha_11=0;

float sum=0;
int start_x=0;
int start_y=0;
int m=0,n=0;
int tid1 = threadIdx.x+blockIdx.x*blockDim.x;
int tid2= threadIdx.y+blockIdx.y*blockDim.y;

if (tid1 < h_im_height && tid2 < h_im_width)
{
	
	prepNshared1(d_image,h_im_width,h_im_height);
	prepNshared2(d_hyst,h_im_width,h_im_height);

	start_x = h_alpha*tid1;
	start_y = h_alpha*tid2;
	
	if(d_hyst[tid1*h_im_width +tid2] != (float)255 || d_map[tid1 * h_im_width+tid2]==(float)125 || d_maph[tid1*h_im_width+tid2]==(float)125)
	{
				patch3_00 = Nshared1[(threadIdx.x+1-1)][threadIdx.y+1-1];
				patch3_01 = Nshared1[(threadIdx.x+1-1)][threadIdx.y+1];
				patch3_02 = Nshared1[(threadIdx.x+1-1)][threadIdx.y+2];
				patch3_10 = Nshared1[(threadIdx.x+1)][threadIdx.y+1-1];
				patch3_11 = Nshared1[(threadIdx.x+1)][threadIdx.y+1];
				patch3_12 = Nshared1[(threadIdx.x+1-1)][threadIdx.y+2];
				patch3_20 = Nshared1[(threadIdx.x+2)][threadIdx.y+1-1];
				patch3_21 = Nshared1[(threadIdx.x+2)][threadIdx.y+1];
				patch3_22 = Nshared1[(threadIdx.x+2)][threadIdx.y+2];	
			
		if(( tid1-1)>=0 && (tid1-1)<h_im_height && (tid2 -1)<h_im_width && (tid2-1)>=0)
		{
			if(Nshared2[threadIdx.x+1-1][threadIdx.y+1-1]==255){
			patch3_00= patch3_11;
			}
		}
		
		if((tid1-1)>=0 && (tid1-1)<h_im_height && (tid2 >= 0) &&( tid2 < h_im_width))
		{
			if(Nshared2[threadIdx.x+1-1][threadIdx.y+1]==255){
			patch3_01= patch3_11;
			}
		}

		if((tid1-1)>=0 &&(tid1-1)<h_im_height && (tid2+1)>=0 && (tid2+1)<h_im_width)
		{
			if(Nshared2[threadIdx.x+1-1][threadIdx.y+2]==255){
			patch3_02=patch3_11;
			}
		}

		if((tid1 >=0)&& (tid1)<h_im_height && (tid2-1)>=0 && (tid2-1) <h_im_width)
		{
			if(Nshared2[threadIdx.x+1][threadIdx.y+1-1]==255){
			patch3_10=patch3_11;
			}
		}

		if((tid1 >= 0)&& (tid1 <h_im_height) && (tid2 >=0) && (tid2 < h_im_width))
		{
			if(Nshared2[threadIdx.x+1][threadIdx.y+1]==255){
			patch3_11=patch3_11;
			}
		}

		if((tid1 >=0) && (tid1 < h_im_height) && (tid2+1)>=0 && (tid2+1) <h_im_width)
		{
			if(Nshared2[threadIdx.x+1][threadIdx.y+2]==255){
			patch3_12=patch3_11;
			}
		}

		if((tid1+1)>=0 && (tid1+1)<h_im_width && (tid2-1)>=0 && (tid2 -1)<h_im_width)
		{
			if(Nshared2[threadIdx.x+2][threadIdx.y+1-1]==255){
			patch3_20= patch3_11;
			}
		}

		if((tid1+1)>=0 && (tid1+1)<h_im_height && (tid2>=0) && (tid2<h_im_width))
		{
			if(Nshared2[threadIdx.x+2][threadIdx.y+1]==255){
			patch3_21= patch3_11;
			}
		}

		if((tid1+1)>=0 && (tid1+1)<h_im_height && (tid2+1)>=0 && (tid2+1)<h_im_width)
		{
			if(Nshared2[threadIdx.x+2][threadIdx.y+2]==255){
			patch3_22=patch3_11;
			}
		}
			
		if(( tid1-1)>=0 && (tid1-1)<h_im_height && (tid2 -1)<h_im_width && (tid2-1)>=0)
		{
			if(Nshared2[threadIdx.x+1-1][threadIdx.y+1-1]==255){
			patch3_00= (patch3_10 + patch3_01)/2;
			}
		}
	
		if((tid1-1)>=0 && (tid1-1)<h_im_height && (tid2 >= 0) &&( tid2 < h_im_width))
		{
			if(Nshared2[threadIdx.x+1-1][threadIdx.y+1]==255){
			patch3_01=(patch3_11 +patch3_01)/2;
			}
		}

		
		if((tid1-1)>=0 &&(tid1-1)<h_im_height && (tid2+1)>=0 && (tid2+1)<h_im_width)
		{
			if(Nshared2[threadIdx.x+1-1][threadIdx.y+2]==255){
			patch3_02=(patch3_12 +patch3_01)/2;
			}
		}

		if((tid1 >=0)&& (tid1)<h_im_height && (tid2-1)>=0 && (tid2-1) <h_im_width)
		{
			if(Nshared2[threadIdx.x+1][threadIdx.y+1-1]==255){
			patch3_10=(patch3_10 +patch3_11)/2;
			}
		}

		if((tid1 >= 0)&& (tid1 <h_im_height) && (tid2 >=0) && (tid2 < h_im_width))
		{
			if(Nshared2[threadIdx.x+1][threadIdx.y+1]==255){
			patch3_11= (patch3_11 + patch3_11)/2;
			}
		}

		if((tid1 >=0) && (tid1 < h_im_height) && (tid2+1)>=0 && (tid2+1) <h_im_width)
		{
			if(Nshared2[threadIdx.x+1-1][threadIdx.y+2]==255){
			patch3_12=(patch3_12 +patch3_11)/2;
			}
		}

		
		if((tid2+1)>=0 && (tid1+1)<h_im_width && (tid2-1)>=0 && (tid2 -1)<h_im_width)
		{
			if(Nshared2[threadIdx.x+2][threadIdx.y+1-1]==255){
			patch3_20=(patch3_10 +patch3_21)/2;
			}
		}

		if((tid1+1)>=0 && (tid1+1)<h_im_height && (tid2>=0) && (tid2<h_im_width))
		{
			if(Nshared2[threadIdx.x+2][threadIdx.y+1]==255){
			patch3_21=(patch3_11 +patch3_21)/2;
			}
		}


		if((tid1+1)>=0 && (tid1+1)<h_im_height && (tid2+1)>=0 && (tid2+1)<h_im_width)
		{
			if(Nshared2[threadIdx.x+2][threadIdx.y+2]==255){
			patch3_22=(patch3_12 +patch3_21)/2;
			}
		}
		
patcha_00 = (patch3_00 +patch3_01+patch3_10+patch3_11)/4;
patcha_01=(patch3_01 +patch3_02+patch3_11+patch3_12)/4;
patcha_10 =(patch3_10+patch3_11+patch3_20+patch3_21)/4;
patcha_11 =(patch3_11+patch3_12+patch3_21+patch3_22)/4;			

sum = (patcha_00 +patcha_01+patcha_10+patcha_11);

patcha_00 = (patcha_00/sum)*h_alpha*h_alpha*d_image[tid1*h_im_width + tid2];
patcha_01=(patcha_01/sum)*h_alpha*h_alpha*d_image[tid1*h_im_width +tid2];
patcha_10= (patcha_10/sum)*h_alpha*h_alpha*d_image[tid1*h_im_width +tid2];
patcha_11=(patcha_11/sum)*h_alpha*h_alpha*d_image[tid1*h_im_width +tid2];

d_high_res[(start_x)*h_alpha*h_im_width+(start_y)]=patcha_00;
d_high_res[(start_x)*h_alpha*h_im_width+(start_y+1)]=patcha_01;
d_high_res[(start_x+1)*h_alpha*h_im_width+start_y]=patcha_10;
d_high_res[(start_x+1)*h_alpha*h_im_width+(start_y+1)]=patcha_11;
	
}

	else if(d_hyst[tid1*h_im_width+tid2]==(float)255)
	{
		sum =0;
		for(m=0;m<h_alpha;m++)
		{
			for(n=0;n<h_alpha;n++)
			{
				sum+=d_high_res[(start_x+m)*h_alpha*h_im_width+(start_y+n)];
			}
		}
		for(m=0;m<h_alpha;m++)
		{
			for(n=0;n<h_alpha;n++)
			{
				d_high_res[(start_x+m)*h_alpha*h_im_width +(start_y +n)]=(d_high_res[(start_x+m)*h_alpha*h_im_width+(start_y+n)]/sum)*(h_alpha*h_alpha)*d_image[tid1*h_im_width+tid2];
			}
		}
	}
	
  }
}



//commbining hrh and hrv

__global__
void gpu_combineHRH_HRV(float *d_high_res,float *d_HRH,float *d_maph,float *d_map,int h_alpha,int h_im_height,int h_im_width)
{

int m=0,n=0;
int tid1 = threadIdx.x+blockIdx.x*blockDim.x;
int tid2= threadIdx.y+blockIdx.y*blockDim.y;

if(tid1 < h_im_height && tid2 < h_im_width)
{
	for(m=0;m<h_alpha;m++)
	{
		for(n=0;n<h_alpha;n++)
		{	
			if(((tid1*h_alpha+m) <h_alpha*h_im_height) &&((tid2*h_alpha+n)*h_alpha*h_im_width))
			{
				d_high_res[(tid1*h_alpha+m)*h_im_width*h_alpha +(tid2*h_alpha+n)]+=d_HRH[(tid1*h_alpha+m)*h_im_width*h_alpha+(tid2*h_alpha+n)];
				if((d_maph[tid1*h_im_width+tid2] == d_map[tid1*h_im_width+tid2]) && d_maph[tid1*h_im_width +tid2]==255)
				{
					d_maph[tid1*h_im_width+tid2]=125;
					d_map[tid1*h_im_width+tid2]=125;
				}
				else if((d_maph[tid1*h_im_width+tid2]==255 && d_map[tid1*h_im_width+tid2] !=255) || (d_maph[tid1*h_im_width+tid2] != 255 && d_map[tid1*h_im_width+tid2]==255))	
				{
					d_high_res[(tid1*h_alpha+m)*h_im_width*h_alpha+(tid2*h_alpha+n)] = d_high_res[(tid1*h_alpha+m)*h_im_width*h_alpha+(tid2*h_alpha+n)]+ d_HRH[(tid1*h_alpha+m)*h_im_width*h_alpha+(tid2*h_alpha+n)];
				}
			}
		}
	}
}


}




void consolidated_convolveMagAngleSuppressionSortVHM(float *vmap,float *hmap,float *h_image,int h_im_width,int h_im_height,float *h_gx_mask,float *h_gy_mask,int h_m_w,float *h_I_angle,float *h_hyst,float *h_gxy,int h_alpha,float *h_HRV,float *h_HRH,float *h_highres){

float *d_image,*d_Gxy_outimage,*d_suppressed,*d_Igx_mask,*d_Igy_mask,*d_I_angle,*d_hyst,*d_HRV,*d_map,*d_HRH,*d_maph,*d_high_res;

//Allocate memory for device pointers
cudaMalloc((void **)&d_I_angle,sizeof(float)*h_im_width*h_im_height);
cudaMalloc((void **)&d_Gxy_outimage,sizeof(float)*h_im_width*h_alpha*h_im_height*h_alpha);
cudaMalloc((void **)&d_image,sizeof(float)*h_im_width*h_im_height);
cudaMalloc((void **)&d_suppressed,sizeof(float)*h_im_width*h_im_height);
cudaMalloc((void **)&d_Igx_mask,sizeof(float)*h_m_w*h_m_w);
cudaMalloc((void **)&d_Igy_mask,sizeof(float)*h_m_w*h_m_w);

//memcpy to device pointers
cudaMemcpy(d_image,h_image,sizeof(float)*h_im_width*h_im_height,cudaMemcpyHostToDevice);
cudaMemcpy(d_Igx_mask,h_gx_mask,sizeof(float)*h_m_w*h_m_w,cudaMemcpyHostToDevice);
cudaMemcpy(d_Igy_mask,h_gy_mask,sizeof(float)*h_m_w*h_m_w,cudaMemcpyHostToDevice);

dim3 dimGrid(ceil((float)h_im_height/(float)BLOCKWIDTH),ceil((float)h_im_width/(float)BLOCKWIDTH),1);
dim3 dimBlock(16,16,1);
gpu_conv_mag_phase<<<dimGrid,dimBlock>>>(d_image,h_im_width,h_im_height,d_Gxy_outimage,d_Igx_mask,d_Igy_mask,d_I_angle,d_suppressed);

gpu_suppress<<<dimGrid,dimBlock>>>(d_Gxy_outimage,d_suppressed,d_I_angle,h_im_width,h_im_height);

cudaFree(d_Gxy_outimage);
cudaFree(d_Igx_mask);
cudaFree(d_Igy_mask);


thrust::device_ptr<float> thr_d(d_suppressed);
thrust::device_vector<float>d_supp_vec(thr_d,thr_d+(h_im_height*h_im_width));
thrust::host_vector<float>h_supp_vec(h_im_height*h_im_width);
thrust::sort(d_supp_vec.begin(),d_supp_vec.end());

cudaMalloc((void **)&d_hyst,sizeof(float)*h_im_width*h_im_height);
cudaMalloc((void **)&d_HRV,sizeof(float)*(h_im_width*h_alpha)*(h_im_height*h_alpha));
cudaMalloc((void **)&d_HRH,sizeof(float)*(h_im_width*h_alpha)*(h_im_height*h_alpha));
cudaMalloc((void **)&d_map,sizeof(float)*h_im_width*h_im_height);
cudaMalloc((void **)&d_maph,sizeof(float)*h_im_width*h_im_height);
cudaMalloc((void **)&d_high_res,sizeof(float)*(h_im_width *h_alpha)*(h_im_width*h_alpha));

cudaMemset(d_high_res,0,sizeof(float)*h_im_width*h_alpha*h_im_width*h_alpha);
cudaMemset(d_HRV,0,sizeof(float)*(h_im_width*h_alpha)*(h_im_height*h_alpha));
cudaMemset(d_HRH,0,sizeof(float)*(h_im_width*h_alpha)*(h_im_height*h_alpha));
cudaMemset(d_map,0,sizeof(float)*(h_im_width)*(h_im_height));
cudaMemset(d_maph,0,sizeof(float)*(h_im_width)*(h_im_height));


//getting high and low threshold

float index = float(0.90)*h_im_height*h_im_width;
float th_high = d_supp_vec[(int)index];
float th_low =th_high/5;

//Double-thresholding
gpu_doubleThreshold<<<dimGrid,dimBlock>>>(d_hyst,d_suppressed,h_im_width,h_im_height,th_high,th_low);

//Edge linking
gpu_edgeLinking<<<dimGrid,dimBlock>>>(d_suppressed,d_hyst,h_im_width,h_im_height);

//Vertical edge keeping
dim3 dimGrid1(1,ceil((float)h_im_width/(float)BLOCKWIDTH),1);
dim3 dimBlock1(1,256,1);
gpu_verticalEdgeKeeping<<<dimGrid1,dimBlock1>>>(d_hyst,d_I_angle,d_image,h_im_width,h_im_height,h_alpha,d_HRV,d_map);


//Horizontal edge keeping
dim3 dimGrid2(ceil((float)h_im_width/(float)BLOCKWIDTH),1,1);
dim3 dimBlock2(512,1,1);
gpu_horizontalEdgeKeeping<<<dimGrid2,dimBlock2>>>(d_hyst,d_I_angle,d_image,h_im_width,h_im_height,h_alpha,d_HRH,d_maph);


d_high_res=d_HRV;
gpu_combineHRH_HRV<<<dimGrid,dimBlock>>>(d_high_res,d_HRH,d_maph,d_map,h_alpha,h_im_height,h_im_width);

gpu_meanKeeping<<<dimGrid,dimBlock>>>(d_image,d_hyst,h_im_width,h_im_height,h_alpha,d_map,d_maph,d_high_res);

cudaMemcpy(h_highres,d_high_res,sizeof(float)*(h_im_width *h_alpha)*(h_im_height * h_alpha),cudaMemcpyDeviceToHost);


cudaFree(d_image);
cudaFree(d_I_angle);
cudaFree(d_suppressed);
cudaFree(d_hyst);
cudaFree(d_HRV);
cudaFree(d_map);
cudaFree(d_HRH);
cudaFree(d_maph);

}

