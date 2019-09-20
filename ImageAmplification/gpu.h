/*************************************************
 *gpu.h is a header file to provide necessary
 *definitions for the BLOCKWIDTH, mask size,
 *apron size and value of PI,etc that are used
 *in GPGPU implementation
*************************************************/
#define BLOCKWIDTH 16
#define g_w 11
#define apron 3
#define BUFFER 512
#define PI 3.14159

void consolidated_convolveMagAngleSuppressionSortVHM(float *vmap,float *hmap,float *h_image,int h_im_width,int h_im_height,float *h_gx_mask,float *h_gy_mask,int h_m_w,float *h_I_angle,float *h_hyst,float *h_gxy,int h_alpha,float *h_HRV,float *h_HRH,float *h_highres);
