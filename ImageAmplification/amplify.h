#include "stdio.h"
#include "math.h"
#include "image_template.h"
#define E 2.71828
#include <vector>
#define PI 3.14159

void print_matrix(float *matrix,int width,int height)
{       
        int i,j;
        for(i=0;i<height;i++)
        {
                printf("\n");
                for(j=0;j<width;j++)
                        printf("%f ",matrix[i*width+ j]);
        }
}
//Check logic
void  mean_keeping(float *org_img,float *Hysteresis,int width,int height,int alpha,float *Vmap,float *Hmap,float *HR)
{

	int i,j,m,n,p,q;

	float *patch3x3;

	float *patchaxa;
	
	float sum=0;

	int start_x,start_y;

	patch3x3 = (float *)malloc(sizeof(float)*(alpha+1)*(alpha+1));	
	
	patchaxa = (float *)malloc(sizeof(float)*alpha*alpha);	
	
	for(i=0;i<height;i++)
		for(j=0;j<width;j++)
		{	
			start_x=alpha*i;
			start_y=alpha*j;
			//Initialize patch pixels to 0
			for (m=0;m<3*3;m++)
				patch3x3[m]=(float)0;

			for(m=0;m<alpha*alpha;m++)
				patchaxa[m]=(float)0;


			if(Hysteresis[i*width + j] !=(float)255 || Vmap[i*width+j]==(float)125 || Hmap[i*width+j]==(float)125) //for a non-pixel
			{
				//Rule 1: extract 3x3 patch of LR. I(x,y) is at patch3x3(1,1)
				for(m=0;m<3;m++)
					for(n=0;n<3;n++)
						if((i-1+m)>=0 && (i-1+m) <height && (j-1+n)>=0 && (j-1+n)<width)
							patch3x3[m*3 +n] = (float)org_img[(i-1+m)*width+(j-1+n)];
			
				if(i==236 && j==105)
				{
//					printf("\n original 3x3 \n");
//					print_matrix(patch3x3,3,3);
				}
			

				for(m=-1;m<=1;m++) //check for edge
					for(n=-1;n<=1;n++)
					{
						if(i+m>=0 && i+m<height && j+n>=0 && j+n<width && m!=n && m*n==0)
							if(Hysteresis[(i+m)*width + (j+n)]==(float)255)
								patch3x3[(1+m)*3+(1+n)]=(float)org_img[i*width + j];
					}

				//Rule 2
				 for(m=-1;m<=1;m++) //check for edge
                                        for(n=-1;n<=1;n++)
                                        {       
                                                if(i+m>=0 && i+m<height && j+n>=0 && j+n<width && m!=0 &&n!=0)
                                                        if(Hysteresis[(i+m)*width + (j+n)]==(float)255)
                                                                patch3x3[(1+m)*3+(1+n)]=(float)(patch3x3[1*3 + (1+n)]+patch3x3[(1+m)*3+1])/2;
                                         
					}			
				//Interpolate
				 sum=0;
				for(m=0;m<alpha;m++)
					for(n=0;n<alpha;n++)
					{	sum=0;
						for(p=0;p<=1;p++)
							for(q=0;q<=1;q++)
								sum+=(float)patch3x3[(m+p)*3 + (n+q)];
						patchaxa[m*alpha+n]=(float)sum/4;
					}						
					//Modify patchaxa
					sum=0;
					for(m=0;m<alpha;m++)
						for(n=0;n<alpha;n++)
							sum+=(float)patchaxa[m*alpha+n];			
					for(m=0;m<alpha;m++)
						for(n=0;n<alpha;n++)
							patchaxa[m*alpha+n]=(float)((patchaxa[m*alpha+n]/sum)*alpha*alpha)*(float)org_img[i*width + j];
		
					//Copy to HR
					for(m=0;m<alpha;m++)
						for(n=0;n<alpha;n++)
							HR[(start_x+m)*alpha*width +(start_y+n)]=(float)patchaxa[m*alpha+n];	
			
			}
			//mean keeping for all pixels
			else if(Hysteresis[i*width+j]==(float)255)
			{
				sum=0;
				for(m=0;m<alpha;m++)
                        		for(n=0;n<alpha;n++)
						sum+=HR[(start_x+m)*alpha*width +(start_y+n)];
	
				for(m=0;m<alpha;m++)
                        	        for(n=0;n<alpha;n++)
						HR[(start_x+m)*alpha*width +(start_y+n)]=(float)((HR[(start_x+m)*alpha*width +(start_y+n)]/sum)*alpha*alpha)*(float)org_img[i*width + j];
			}
	/*		if(i==236 && j==105)
			{
				printf("\n paxa \n");
				print_matrix(patchaxa,2,2);
				
				printf("\n patch3x3 \n");
				print_matrix(patch3x3,3,3);

				printf("\n pixel intensity: %f\n",(float)org_img[i*width+j]);
			}*/
	}
} 

int compare(int v,int d, int ad)
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

void horizontal_edge_keeping(float *Hysteresis,float *I_angle,float *org_img,int width,int height,int alpha,float *HR,float *map)
{
	int i,k,m,n;

	std::vector <int> rod_x;
	std::vector <int> rod_y;
	int horizontal,diagonal,anti_diagonal;
        int horizontal_start=1;
        float theta=0;
        int direction=0;
        int start_x,start_y,end_x,end_y; //Start and stop positions for filling in the HR Image
	int count=0;
	float ND=0;
	int rodysize=0,rodxsize=0;

	for(i=0;i<height;i++)
	{
		horizontal_start=0;
		while(horizontal_start<width) //not to exceed the width of the image
		{
			//go to the start of the horizontal rod
			while(Hysteresis[i*width+horizontal_start]==0 && horizontal_start<width)
				horizontal_start++;

			while(Hysteresis[i*width + horizontal_start]!=0 && horizontal_start<width)
			{
				rod_x.push_back(i);
				rod_y.push_back(horizontal_start);
				horizontal_start++;
			}
			

		//Check the orientation
	
			horizontal=0;diagonal=0;anti_diagonal=0;
			
			rodysize=rod_y.size();
						
			for(k=0;k<rodysize;k++)
			{
				//Each rod IS a rod
				map[rod_x[k]*width+rod_y[k]]=(float)255;
				theta=I_angle[rod_x[k]*width+rod_y[k]];
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
			direction= compare(horizontal,diagonal,anti_diagonal);
		
			//printf("H: %d D:%d AD:%d \n",horizontal,diagonal,anti_diagonal);
			ND=0;
			if(horizontal==0 && diagonal==0 && anti_diagonal==0)
			{
			//	printf("\n unkown direction for H rod. Setting to 0");
				ND=1;
			}
		
			rodysize=rod_y.size();		

			if(ND==1)
				for(k=0;k<rodysize;k++)
					map[rod_x[k]*width+rod_y[k]]=(float)125;
		//Checkpoint-1
		
			//printf("\n The %d rod (size: %d) starts at (%d,%d) and ends at (%d,%d). Orientation:%d",rod_count++,rod_x.size(),rod_x.front(),rod_y.front(),rod_x.back(),rod_y.back(),direction);

			if(direction==0 && ND==0) //replicate horizontally
			{
				start_x=alpha*rod_x.front();
				start_y=alpha*rod_y.front();
				
				end_x=alpha*rod_x.back();
				end_y=alpha*rod_y.back();
		
				count=0;
				rodxsize=rod_x.size();
				rodysize=rod_y.size();

				for(k=0;k<alpha;k++) //no. of times to replicate
				{
					
					for(m=0;m<rodxsize;m++)
					{			
						if(start_x<alpha*height && (start_y+k*rodysize+m)<alpha*width)
						//HR[(start_x*alpha*width) + (start_y+k*rod_y.size()+m)]=(float)255;	
							HR[(start_x*alpha*width) + (start_y+k*rodysize+m)]=(float)org_img[rod_x[m]*width + rod_y[m]];
						//Bottom pixels need to be filled
						for(n=1;n<=(alpha-1);n++)
							if((start_x+n)<alpha*height && (start_y+k*rodysize+m)<width && (rod_x[m]+1)<height)
								HR[(start_x+n)*alpha*width + (start_y+k*rodysize+m)]=(float)org_img[(rod_x[m]+1)*width + rod_y[(int)(count/alpha)]];

						count++;		
					}
				}
			}
			else if(direction==1 && ND==0) //diagonal direction, replicate upwards
			{

				start_x=alpha*rod_x.front() + direction*(alpha-1);
				start_y=alpha*rod_y.front();

				rodxsize=rod_x.size();
                                rodysize=rod_y.size();				

				count=0;
				for(k=0;k<alpha;k++)
				{
					for(m=0;m<rodysize;m++)
					{
						if((start_x -1*k*direction)<alpha*height && (start_x -1*k*direction)>=0 && (start_y + k*rodysize+m)<alpha*width && (start_y + k*rodysize+m)>=0) 
				//			HR[(start_x -1*k*direction)*alpha*width + (start_y + k*rod_y.size()+m)]=(float)255;	
							HR[(start_x -1*k*direction)*alpha*width + (start_y + k*rodysize+m)]=(float)org_img[rod_x[m]*width + rod_y[m]];
						//Top pixels need to be filled if k==0
						if(k==0)
							for(n=1;n<=(alpha-1);n++)
								if((start_x-1*k*direction-n)>=0 && (start_x-1*k*direction-n)<alpha*height && (start_y+k*rodysize+m)<width && (rod_x[m]-1)>=0)
									HR[(start_x-1*k*direction-n)*alpha*width + (start_y+k*rodysize+m)]=(float)org_img[(rod_x[m]-1)*width + rod_y[(int)(count/alpha)]];
								
						else //fill the bottom pixels
							for(n=1;n<=(alpha-1);n++)        
								if((start_x-1*k*direction+n)<alpha*height && (start_x-1*k*direction+n)>=0 && (start_y+k*rodysize+m)<width && (rod_x[m]+1)<height)
                                                                        HR[(start_x-1*k*direction+n)*alpha*width + (start_y+k*rodysize+m)]=(float)org_img[(rod_x[m]+1)*width + rod_y[(int)(count/alpha)]];	
	
						count++;
					}
				}

			}
			else if(direction==-1 &&ND==0)//anti-diagonal direction, replicate downwards 
			{
				start_x=alpha*rod_x.front();
				start_y=alpha*rod_y.front();
				count=0;
		

				rodxsize=rod_x.size();
                                rodysize=rod_y.size();

				for(k=0;k<alpha;k++)
				{

					for(m=0;m<rodysize;m++)
					{
						if((start_x -1*k*direction)<alpha*height && (start_x -1*k*direction)>=0 && (start_y + k*rodysize+m)<alpha*width && (start_y + k*rod_y.size()+m)>=0)
		//				HR[(start_x-1*k*direction)*alpha*width + (start_y + k*rod_y.size() + m)]=(float)255;	
							HR[(start_x-1*k*direction)*alpha*width + (start_y + k*rodysize + m)]=(float)org_img[rod_x[m]*width + rod_y[m]];
				
						if(k==0) //fill bottom pixels
							for(n=1;n<=(alpha-1);n++)                                                             
								if((start_x-1*k*direction+n)<alpha*height && (start_x-1*k*direction+n)>=0 && (start_y+k*rodysize+m)<width && (rod_x[m]+1)<height)
                                                	        	HR[(start_x-1*k*direction+n)*alpha*width + (start_y+k*rodysize+m)]=(float)org_img[(rod_x[m]+1)*width + rod_y[(int)(count/alpha)]];


						else //fille the top
							for(n=1;n<=(alpha-1);n++)                                                             
								if((start_x-1*k*direction-n)>=0 && (start_x-1*k*direction-n)<alpha*width && (start_y+k*rodysize+m)<width && (rod_x[m]-1)>=0)
                                                                        HR[(start_x-1*k*direction-n)*alpha*width + (start_y+k*rodysize+m)]=(float)org_img[(rod_x[m]-1)*width + rod_y[(int)(count/alpha)]];
						count++;
					}
				}		

				

			}
		
/**/
			rod_x.erase(rod_x.begin(),rod_x.end());
                	rod_y.erase(rod_y.begin(),rod_y.end());
		}//end horizontal movement
	}//end of i for height

}

void vertical_edge_keeping(float *Hysteresis,float *I_angle,float *org_img,int width,int height,int alpha,float *HR,float *map)
{

	int i,j,k,m,n;

	std::vector <int> rod_x;
	std::vector <int> rod_y;
	int vertical,diagonal,anti_diagonal;
	int rod_count=0;
	int vert_start=1;
	float theta=90;
	int direction=0;
	int count=0;
	int ND=0;
	int start_x,start_y,end_x,end_y; //Start and stop positions for filling in the HR Image
	for(j=0;j<width;j++)
	{	vert_start=0;
		
		while(vert_start<height) //not to exceed the height of the image
		{
			//go to the starting of the rod
			while(Hysteresis[vert_start*width + j]==0 && vert_start<height)
				vert_start++;
			
			while(Hysteresis[vert_start*width + j]!=0 && vert_start < height)
			{
				rod_x.push_back(vert_start);
				rod_y.push_back(j);
				vert_start++;
				//printf("\n pixel (%d,%d) in rod",vert_start,j);
				//getchar();	
			}
			//Check the orientation
			vertical=0;diagonal=0,anti_diagonal=0;

			for(k=0;k<rod_x.size();k++)
			{
				//Fill the edge map. Each rod IS a rod
				map[rod_x[k]*width + rod_y[k]]=(float)255;
				theta=I_angle[rod_x[k]*width + rod_y[k]];
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
			
			//Fill the 

			direction=compare(vertical,diagonal,anti_diagonal);
			ND=0;
			if(vertical==0 && diagonal==0 &&anti_diagonal==0)
			{
				//printf("\n Unknown direction for V rod. Set to 0");
				ND=1;
			}
			if(ND==1)
				for(k=0;k<rod_x.size();k++)
					map[rod_x[k]*width + rod_y[k]]=(float)125;
	
			//if(!ND)
			//	printf("\n (ND %d) rod %d of size %d starts at (%d,%d) ends at (%d,%d) direction %d.",ND,rod_count++,rod_x.size(),rod_x.front(),rod_y.front(),rod_x.back(),rod_y.back(),direction);

			if(direction == 0 && ND==0) //vertical
			{
				start_x=alpha*rod_x.front();
				start_y=alpha*rod_y.front();
				end_x=alpha*rod_x.back();
				end_y=alpha*rod_y.back();

				count=0;
				for(k=0;k<alpha;k++) //Number of times to repeat this rod
				{	
					for(m=0;m<rod_x.size();m++)
					{
						if((start_x + k*rod_x.size()+m)<alpha*height &&start_y<alpha*width)
							//HR[(start_x + k*rod_x.size()+m)*alpha*width + start_y]=(float)255;
							HR[(start_x + k*rod_x.size()+m)*alpha*width + start_y]=(float)org_img[rod_x[m]*width + rod_y[m]];
						//Pixel to the right need to be filled
						for(n=1;n<=(alpha-1);n++)
							if((start_x + k*rod_x.size()+m)<alpha*height &&(start_y+n)<alpha*width && (rod_y[m]+1)<width) //copy adjacent pixel
								HR[(start_x + k*rod_x.size()+m)*alpha*width + start_y+n]=(float)org_img[rod_x[(int)(count/alpha)]*width + (rod_y[m]+1)]; 
						count++;			
					}
					
				}				
			}
			else if(direction==1 && ND==0) //diagonal
			{

				start_x=alpha*rod_x.front(); 
				start_y=alpha*rod_y.front() + direction*(alpha-1); //the rod direction is perpendicular to the gradient. Start from the right corner
				end_x=alpha*rod_x.back();
				end_y=alpha*rod_y.back();

				count=0;
				for(k=0;k<alpha;k++) //Number of times to repeat this rod
				{
					for(m=0;m<rod_x.size();m++)
					{
						if((start_x + k*rod_x.size()+m)<alpha*height &&(start_y-1*k*direction)<alpha*width&&(start_y-1*k*direction)>=0)
							//HR[(start_x + k*rod_x.size()+m)*alpha*width + (start_y-1*k*direction)]=(float)255;
						HR[(start_x + k*rod_x.size()+m)*alpha*width + (start_y-1*k*direction)]=(float)org_img[rod_x[m]*width + rod_y[m]];
					//if k=0, fill pixels to the left
						if(k==0)
						{
							for(n=1;n<=(alpha-1);n++)                                        
                						if((start_x + k*rod_x.size()+m)<alpha*height &&(start_y-1*k*direction-n)>=0 && (start_y-1*k*direction-n)<alpha*width && (rod_y[m]-1)>=0) //copy adjacent pixel
                                                                	HR[(start_x + k*rod_x.size()+m)*alpha*width + start_y-1*k*direction-n]=(float)org_img[rod_x[(int)(count/alpha)]*width + (rod_y[m]-1)]; 
                                                	count++;
						}	
					//if k=1, fill pixels to the right
						else
						{
							for(n=1;n<=(alpha-1);n++)
                                                        	if((start_x + k*rod_x.size()+m)<alpha*height &&(start_y-1*k*direction+n)<alpha*width && (start_y-1*k*direction+n)>=0 && (rod_y[m]+1)<width) //copy adjacent pixel
                                                                	HR[(start_x + k*rod_x.size()+m)*alpha*width + start_y-1*k*direction+n]=(float)org_img[rod_x[(int)(count/alpha)]*width + (rod_y[m]+1)];
                                                	count++;
						}
					}
				}
			}
			else if(direction==-1 && ND==0)//anti-diagonal
			{

				start_x=alpha*rod_x.front(); 
				start_y=alpha*rod_y.front(); //the rod direction is perpendicular to the gradient. Starts at same y position as vertical
				end_x=alpha*rod_x.back();
				end_y=alpha*rod_y.back();
				count=0;
				for(k=0;k<alpha;k++) //Number of times to repeat this rod
				{
					for(m=0;m<rod_x.size();m++)
					{	if((start_x + k*rod_x.size()+m)<alpha*height &&(start_y-1*k*direction)<alpha*width&&(start_y-1*k*direction)>=0)
							//HR[(start_x + k*rod_x.size()+m)*alpha*width + (start_y-1*k*direction)]=(float)255;				
							HR[(start_x + k*rod_x.size()+m)*alpha*width + (start_y-1*k*direction)]=(float)org_img[rod_x[m]*width + rod_y[m]];

						if(k==0) //fill to the right
						{
							for(n=1;n<=(alpha-1);n++)
                                                        	if((start_x + k*rod_x.size()+m)<alpha*height &&(start_y-1*k*direction+n)<alpha*width && (start_y-1*k*direction+n)>=0 && (rod_y[m]+1)<width) //copy adjacent pixel
                                                                	HR[(start_x + k*rod_x.size()+m)*alpha*width + start_y-1*k*direction+n]=(float)org_img[rod_x[(int)(count/alpha)]*width + (rod_y[m]+1)];
                                                        count++;
						}
						else //fill to the left
						{
							for(n=1;n<=(alpha-1);n++)
                                                                if((start_x + k*rod_x.size()+m)<alpha*height &&(start_y-1*k*direction-n)>=0 && (start_y-1*k*direction-n)<alpha*width && (rod_y[m]-1)>=0) //copy adjacent pixel
                                                                        HR[(start_x + k*rod_x.size()+m)*alpha*width + start_y-1*k*direction-n]=(float)org_img[rod_x[(int)(count/alpha)]*width + (rod_y[m]-1)];
                                                        count++;
						}
					}
				}
			}

			//printf("\n The %d rod (size: %d) starts at (%d,%d) and ends at (%d,%d). Orientation:%d",rod_count++,rod_x.size(),rod_x.front(),rod_y.front(),rod_x.back(),rod_y.back(),compare(vertical,diagonal,anti_diagonal));
			rod_x.erase(rod_x.begin(),rod_x.end());	
			rod_y.erase(rod_y.begin(),rod_y.end());	
		//getchar();
		} 
	}//end for j on width

}
