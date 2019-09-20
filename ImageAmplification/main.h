#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <cstring>

typedef enum{ FALSE, TRUE} boolean;

/*Function that returns a gaussian mask in the X direction*/
void gaussianX_mask(float *mask, int gauss_width, float sigma)
{
	int i,j;
	float E = 2.7182818;
	
	for(i=-1*gauss_width/2; i <= gauss_width/2; i++)
	{
		for(j=-1*gauss_width/2; j <= gauss_width/2; j++)
		{
			mask[(i+gauss_width/2)*gauss_width+j+gauss_width/2]= -1*i*pow(E,-1*(i*i+j*j)/(2*sigma*sigma));
		}
	}
}
/*   Function that returns a gaussian mask in the Y direction  */
void gaussianY_mask(float *mask, int gauss_width, float sigma)
{
	int i,j;
	float E = 2.7182818;
	
	for(i=-1*gauss_width/2; i <= gauss_width/2; i++)
	{
		for(j=-1*gauss_width/2; j <= gauss_width/2; j++)
		{
			mask[(i+gauss_width/2)*gauss_width+j+gauss_width/2]= -1*j*pow(E,-1*(i*i+j*j)/(2*sigma*sigma));
		}
	}
}
/*  Define and declare convolve function  */
template <class T>
void convolve(T *org_img, int img_width, int img_height, float *mask, int gauss_width, float *op_img)
{
	int i, j, k, l;
	float temp;
	for(i=0; i<img_height; i++)
	{
		for(j=0; j<img_width; j++)
		{
			temp = 0;
			for(k=-1*gauss_width/2; k<=gauss_width/2; k++)
			{
				for(l=-1*gauss_width/2; l<=gauss_width/2; l++)
				{
					if(((i+k) < img_height) && ((j+l) < img_width) && ((i+k)>= 0) && ((j+l)>=0))
						temp += mask[(k+gauss_width/2)*gauss_width+(l+gauss_width/2)]*org_img[(i+k)*img_width+(j+l)];
				}
			}
			op_img[i*img_width+j] = (float)temp;
		}
	}
}
/*   Define and declare convolve function with an offset   */
template <class T> void convolve2(T *org_img, float *mask, float *op_img, int img_width, int img_height, int gauss_width, int offset, int padHpc)
{
        float temp;
        int i, j, k, l;

        for(i = offset; i < img_height + offset; i++)
        {
                for(j = 0; j < img_width; j++)
                {
                        temp = 0;
                        for(k = -1*gauss_width/2; k <= gauss_width/2; k++)
                        {

                                for(l = -1*gauss_width/2; l <= gauss_width/2; l++)
                                {
                                        if((i+k) < padHpc && (j+l) < img_width && (i+k) >=0 && (j+l) >= 0)
                                        {
                                                temp += mask[gauss_width*(k+gauss_width/2) + l+gauss_width/2]*org_img[(i+k)*img_width + j+l];
                                        }
                                }
                        }
                        op_img[(i - offset)*img_width + j] = (T)temp;
                }
        }
}
/*   Goes pixel by pixel and calculates the magnitude of the pixel    */
void magnitude(float *mask1, float *mask2, int img_width, int img_height, float *Gxy)
{
	int i, j;
	float temp1, temp2;
	for(i=0; i<img_height; i++)
	{
		for(j=0; j<img_width; j++)
		{
			temp1 = mask1[i*img_width+j] * mask1[i*img_width+j];
			temp2 = mask2[i*img_width+j] * mask2[i*img_width+j];
			Gxy[i*img_width+j] = sqrt(temp1+temp2);
		}
	}
}
/*   Goes pixel by pixel and calculates the directionality of the pixel    */
void angle(float *gy_img, float *gx_img, int img_width, int img_height, float *I_angle)
{
	int i, j;
	for(i=0; i<img_height; i++)
	{
		for(j=0; j<img_width; j++)
		{
			I_angle[i*img_width+j] = atan2(gy_img[i*img_width+j], gx_img[i*img_width+j]);
		}
	}
}

//Needed for qsort functionality
int comp(const void * a, const void * b)
{
	return ( *(int*)a - *(int*)b );
}
void double_thresh(float *suppress, float * hyst, int img_width, int img_height)
{
	int i, j, index;
	float thresh, thresh_low;
	float *supp;

	supp = (float *)malloc(sizeof(float) * (img_width * img_height));
	memcpy(supp, suppress, sizeof(float)*img_width * img_height); //VKP: You forgot include sizeof(float). memcpy is byte-wise and doesn't care about the data-type
 
 	memcpy(hyst, suppress, sizeof(float)*img_width * img_height); //VKP: this copy operation was missing

	qsort(supp, img_width * img_height, sizeof(float), comp);
	
	index = img_width * img_height * float(0.90); //VKP: Change it to 0.90
	
	thresh = supp[index];
	
	thresh_low = thresh/5;
	
	//Assign 255 to strong pixels and 125 to weak pixels
	
	for(i=0;i<img_height;i++)
                for(j=0;j<img_width;j++)
                        if(hyst[i*img_width+j]>thresh)
                                hyst[i*img_width + j]= (float)255;
                        else if (hyst[i*img_width+j]<thresh && hyst[i*img_width+j]>thresh_low)
                                hyst[i*img_width + j]= (float)125;
                        else
                                hyst[i*img_width + j]= (float)0;	
		
}

void edge_linking(float *buffer, float *A , int width, int height)
{

	int i,j; //variables were undeclared
	int buff_height=height;

	int offset=0;
	memcpy(A,buffer,sizeof(float)*width*height); //VKP: This memcpy was missing too

	for(i=offset;i<height+offset;i++)
                for(j=0;j<width;j++)
                {

                        if(buffer[i*width + j]==(float)125) //Check nearest 8 neighbors
                        {
                //                flag=1;
				//initially set the weak pixel to 0     
				A[(i-offset)*width + j]=(float)0;	

				//top and bottom (i-1,j) and (i+1,j)
				 if(i-1>=0)
                                        if(buffer[(i-1)*width +j]==(float)255)
                                                A[(i-offset)*width+j]=(float)255;
                                if(i+1<buff_height)
                                        if(buffer[(i+1)*width + j]==(float)255)
                                                A[(i-offset)*width+j]=(float)255;
				
				//left and right (i,j-1) and (i,j+1)
				 if(j-1>=0)
                                        if(buffer[i*width + j-1]==(float)255)
                                                A[(i-offset)*width + j]=(float)255;
                                if(j+1<width)
                                        if(buffer[i*width + j+1]==255)
                                                 A[(i-offset)*width + j]=(float)255;
				
				//Diagonal Pixels (i-1,j-1) and (i+1,j+1)
				if(i-1 >=0 && j-1 >=0)
                                        if(buffer[(i-1)*width + j-1]==(float)255)
                                                A[(i-offset)*width + j]=(float)255;
                                if(i+1<buff_height && j+1 <width)
                                        if(buffer[(i+1)*width + j+1]==(float)255)
                                                A[(i-offset)*width + j]=(float)255;

				//Anti-Diagonal Pixels (i-1,j+1) (i+1,j-1)
				if(i-1>=0 && j+1 <width)
                                        if(buffer[(i-1)*width + j+1]==(float)255)
                                                A[(i-offset)*width + j]=(float)255;
                                if(i+1<buff_height && j-1 >=0)
                                        if(buffer[(i+1)*width + j-1]==(float)255)
                                                A[(i-offset)*width + j]=(float)255;
                        }
                }

				
}



//VKP: Dr. Pallipuram's version of suppression
template <class T>
void nonmaximal_suppression(float *img,float *angle,int offset, int width,int height)
{
        int i,j;
        float theta=0;int count=0;
        float mag=0;
        for(i=offset;i<offset+height;i++)
        {
                for(j=0;j<width;j++)
                {
                        theta=angle[i*width+j];
                        //convert theta from radians to degrees
                        theta=(180/M_PI)*theta;
                        if(theta<0)
                                theta+=(float)180;
                        mag=img[i*width+j];
                        if(theta>(157.5) || theta <=22.5) //Left and Right
                        {
                                if((j-1) >=0)
                                  if(mag<img[(i)*width+j-1])
                                              img[i*width+j]=(float)0;

                                if((j+1)<width)
                                        if(mag<img[(i)*width+j+1])
                                                img[i*width+j]=(float)0;
                        }

                        if(theta>(22.5) && theta <=67.5) //Diagonal pixels
                        {
                                if((i-1)>=0 && (j-1) >=0)
                                        if(mag<img[(i-1)*width+j-1])
                                                img[i*width+j]=(float)0;
                                if((i+1)<height && (j+1) <width)
                                        if(mag<img[(i+1)*width+j+1])
                                                img[i*width+j]=(float)0;
                        }
                        if(theta>(67.5) && theta <= 112.5) //top and bottom. 
                        {
                                if((i-1)>=0)
                                        if(mag<img[(i-1)*width+j])
                                                img[i*width+j]=(float)0;
                                if((i+1)<height)
                                        if(mag<img[(i+1)*width+j])
                                                img[i*width+j]=(float)0;

                        }
                        if(theta>(112.5) && theta <= 157.5) //Anti Diagonal.
                        {
                                if((i+1)<height && (j-1)>=0)
                                        if(mag<img[(i+1)*width + j-1])
                                                img[i*width+j]=(float)0;
                                if((i-1) >=0 &&(j+1)<=width)
                                        if(mag<img[(i-1)*width+j+1])
                                                img[i*width+j]=(float)0;
                                                img[i*width+j]=(float)0;


                        }
                }
        }
}


