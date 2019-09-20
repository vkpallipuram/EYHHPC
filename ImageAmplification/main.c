/********************************************************************************
 *Description : main.c invokes the image amplification process on GPGPU. 
 *Image is read and is input to the function to invoke on GPGPU.
********************************************************************************/

/*Header Files */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "main.h"
#include "amplify.h"
#include "gpu.h"

int main (int argc, char **argv)
{

	float *org_img;
	float *Gx_mask, *Gy_mask;
	float *IGx, *IGy;	
	float *Gxy, *I_angle, *High_res;
	float *suppressed, *threshold, *hyst;
	float *Vmap, *Hmap;
	float *HRV, *HRH;
	int i, j, m, n;
	int img_width, img_height;
	int alpha;
	int sigma;
	int gauss_width;
	

	char fileHigh[25];

	if(argc != 5)
	{
		printf("\n Correct Usage: ./myexec <image name> <gaussian width> <sigma> <alpha>");
		return 0;
	}
	
	//read the image
        read_image_template<float>(argv[1], &org_img, &img_width, &img_height); 
	
	
	//Read arguments from the command line 
	gauss_width = atoi(argv[2]);
	sigma = atoi(argv[3]);
	alpha = atoi(argv[4]);
	
	
	//Allocate memory for masks
	Gx_mask = (float *)malloc(sizeof(float) * (gauss_width * gauss_width));
	Gy_mask = (float *)malloc(sizeof(float) * (gauss_width * gauss_width));

	//Allocate memory for intermediate images
	//VKP: Do you need these?
	IGy = (float *)malloc(sizeof(float) * (img_width * img_height));
	IGx = (float *)malloc(sizeof(float) * (img_width * img_height));
	Gxy = (float *)malloc(sizeof(float) * (img_width * img_height));
	threshold = (float *)malloc(sizeof(float) * (img_width * img_height));

	//Allocations pertaining to image amplification
	High_res = (float *)malloc(sizeof(float) * ((img_width * alpha) * (img_height * alpha)));
	HRV = (float *)malloc(sizeof(float) * ((img_width * alpha) * (img_height * alpha)));
	HRH = (float *)malloc(sizeof(float) * ((img_width * alpha) * (img_height * alpha)));
	
	hyst = (float *)malloc(sizeof(float) * (img_width * img_height));
	I_angle =(float *)malloc(sizeof(float) * (img_width * img_height));
	
	Vmap = (float *)malloc(sizeof(float) * (img_width * img_height));
	Hmap = (float *)malloc(sizeof(float) * (img_width * img_height));

	//Calculate Gaussian masks
	gaussianX_mask(Gx_mask, gauss_width, sigma);
	gaussianY_mask(Gy_mask, gauss_width, sigma);


	consolidated_convolveMagAngleSuppressionSortVHM(Vmap,Hmap,org_img,img_width,img_height,Gx_mask,Gy_mask,gauss_width,I_angle,hyst,Gxy,alpha,HRV,HRH,High_res);
	

	/********************* Lenna no artifacts up to this point********************/
	/* Normalization is needed because some pixel are beyond 255 */
	/* Find highest pixel value in the High Resolution Image */
	float high_val = 0;
	for(i=0;i<img_height; i++)
	{
		for(j=0;j<img_width; j++)
		{
			for(m=0;m<alpha; m++)
			{
				for(n=0;n<alpha;n++)
				{
					if((i*alpha+m)<alpha*img_height && (j*alpha+n) <alpha*img_width)
					{
						if(High_res[(i*alpha+m)*img_width*alpha+(j*alpha+n)] > high_val)
						{
							high_val = High_res[(i*alpha+m)*img_width*alpha+(j*alpha+n)];
						}
					}
				}
			}
		}
	}
	/* Normalize the image */
	for(i=0;i<img_height; i++)
	{
		for(j=0;j<img_width; j++)
		{
			for(m=0;m<alpha; m++)
			{
				for(n=0;n<alpha;n++)
				{
					if((i*alpha+m)<alpha*img_height && (j*alpha+n) <alpha*img_width)
					{
						High_res[(i*alpha+m)*img_width*alpha+(j*alpha+n)] =  High_res[(i*alpha+m)*img_width*alpha+(j*alpha+n)]*(255/high_val);
					}
				}
			}
		}
	}
	
	sprintf(fileHigh, "output%d.pgm", img_width*alpha);
	
	//Write out the images
	write_image_template<float>(fileHigh, High_res, img_width * alpha, img_height * alpha);
	
	free(IGx);
	free(IGy);
	free(Gxy);
	free(I_angle);
	free(hyst);
	free(High_res);
	free(threshold);
	
	return 0;

}
