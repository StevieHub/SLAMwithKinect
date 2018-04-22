#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <math.h>

#define PI 3.14159265

using namespace cv;
using namespace std;

Mat image;
Mat imahe;
Mat show;
Mat depthMap;
Mat bgrImage;
Mat rotim;
Mat areamap;
Mat rotmat;
Mat trilocmap;
Mat buff1;
Mat locmap(450,640,CV_8U,cv::Scalar(255));	//initialize an all-white 640x450 local map image
Mat image1(1000,1000,CV_8U,cv::Scalar(255));	//initialize an all-white 1000x1000 image

std::vector<std::vector<cv::Point> > contours;

Mat rotate(int angle, Mat image){
	int centreX= image.size().width/2;
	int centreY= image.size().height/2;
	int angle1= angle;
	Point2f centre(centreX, centreY);
	rotmat= getRotationMatrix2D(centre, angle1, 1.0);
	warpAffine(image, image1,rotmat,image1.size());
	
	return image1;

}

float mtopixdy(int distance, int angle){
	float pixdy = floor((distance*(abs(cos((angle * PI)/180)))/2)); //5 cm per pixel 
	return pixdy;
}
float mtopixdx(int distance, int angle){
	float pixdx = floor((distance*(abs(sin((angle * PI)/180)))/2)); //5 cm per pixel
	return pixdx;
}

int getquad(int angle){
	int quadval;
	if(angle == 0){
	quadval = 0;
	}
	if(angle > 0 && angle < 90){
	quadval = 1;
	}
	if(angle == 90){
	quadval = 90;
	}
	if(angle > 90 && angle < 180){
	quadval = 2;
	}
	if(angle == 180){
	quadval = 180;
	}
	if(angle > 180 && angle < 270){
	quadval = 3;
	}
	if(angle == 270){
	quadval = 270;
	}
	if(angle > 270){
	quadval = 4;
	}
	return quadval;
}

void editarmap(int y, int x){
	areamap.at<uchar>(y, x) = 255;
}

void editarmaploc(int y, int x){
	areamap.at<uchar>(y, x) = 150;
}


float pix2m (int pixval)
{
	float actval;
	actval = (-0.0168*(pow(pixval,2)) + 10.564*pixval - 5.0894)/5;
	return actval;
}

void editlocmap(int pixval, int x){
	int actval = pix2m(pixval);
	int y = floor(actval);
	locmap.at<uchar>(y, x) = 0;
}

void edittrlocmap(int y, int x){
	trilocmap.at<uchar>(y, x) = 0;
}

int main(){

	VideoCapture capture( CV_CAP_OPENNI );

	int xbuff= 250;
	int ybuff= 250;
	for(int x=0;x<1000;x++)
    {
		char dum;
		cout << "continue? (y/n)" << endl;
		cin >> dum;
		if(dum == 'n'){
			return -1;
		}
	
		capture >> depthMap;
		capture >> bgrImage;

        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {
            if(capture.retrieve( depthMap, CV_CAP_OPENNI_DEPTH_MAP ) )
            {
                const float scaleFactor = 0.06f; 
				depthMap.convertTo( show, CV_8UC1, scaleFactor ); //moomoo! convert to openCV editable format
				imwrite("depthmap.bmp", image);
            }
            if( capture.retrieve( bgrImage, CV_CAP_OPENNI_BGR_IMAGE ) )
	    	{
				resize(bgrImage, buff1, Size(), 0.4,0.4,INTER_LINEAR );
				imwrite("RGBimage.jpg", buff1);		
	    	}
		image = show;
	//========================================================== moo important
		image = image.rowRange(1,475);	//CROP the image to avoid depth map noise
		
		trilocmap = imread("angle.bmp");	
		trilocmap.convertTo(trilocmap,CV_8U);	
		cvtColor(trilocmap, trilocmap, CV_BGR2GRAY);
		Mat logsobel;
		
		Sobel(image,logsobel,CV_8U,0,1,3,80,128); //vertical sobel filter
		medianBlur(logsobel,logsobel,9);

		logsobel.convertTo(logsobel, CV_64F, 1);	//convert image; done for logarithm filter
		log(logsobel, logsobel);

		dilate(logsobel,logsobel,Mat(),Point(-1,-1), 3); //dilate and erode to remove noise
		erode(logsobel,logsobel,Mat(),Point(-1,-1), 5);
			
		logsobel.convertTo(logsobel,CV_8U, 255, 255); //convert the image to 8-bit grayscale for findContours() function
		//cvtColor(logsobel, logsobel, CV_RGB2GRAY);

		findContours(logsobel, contours, // a vector of contours
			CV_RETR_LIST,	// retrieve all contours
			  CV_CHAIN_APPROX_NONE ); // contour approximation: none

		// Eliminate too short contours as these would not be the floor
		unsigned int cmin= 1200; // minimum contour length;
		std::vector<std::vector<cv::Point> > :: iterator i= contours.begin();
		while (i!=contours.end()) {
			if (i->size() < cmin)
			i= contours.erase(i);
			else
			++i;
		}
	

	// Draw black contours on a white image
	Mat result(image.size(),CV_8U,cv::Scalar(255));
	drawContours(result,contours,
		-1, // draw all contours
		cv::Scalar(1), // in black
		-1); // fill it in
	//imwrite("contours.jpg",  result);	
	image= image - result;

	std::vector<std::vector<cv::Point> > polyfill(1,vector<Point> (600));
	Point tmp;
	int pixprev,pixcurrent,actval,actprev=0,consec=0,plotx, fillcount=0,actvalprev=0,fillprev;
	tmp.x = 0;
	tmp.y = locmap.size().height;
	polyfill[0][fillcount] = tmp;
	fillcount++;	
	for( int xit=1; xit<=image.size().width; xit++){
		pixprev=256;
		for(int yit=1; yit<=400; yit++){
			pixcurrent = image.at<uchar>(yit,xit);
			if(pixcurrent>0 && pixcurrent<pixprev)
			{
				pixprev = pixcurrent;
			}
		}

		if (pixprev>=255 && actprev!=0 && consec<5)		//if no data was useful from the processed column
//there are algorithms in place to simply use previous pixel data if no current data is useful. this helps to control noise
		{
			actval = actprev;
			consec++;
		}
		else if(xit<20 || xit>=610)
			actval = pix2m(255);
		else if(consec>=5)
			actval = pix2m(0);
		else
		{
			actval = pix2m(pixprev);
			consec=0;
		}
		plotx = float(xit - (640/2))* (actval+(-10)) * 0.0021 * (640/480);
		plotx = (plotx)+320;
		if((fillprev!=plotx || actval!=actvalprev) && (actvalprev>0) && (fillprev)>0 && (actval<locmap.size().height) && plotx<locmap.size().width){
//^ to prevent segmentation fault
			locmap.at<uchar>(actval,plotx)=0;
			tmp.x = plotx;
			tmp.y = actval;
			polyfill[0][fillcount] = tmp;			
			fillcount++;
		}

		if(pixprev!=256){
			actprev=actval;		//changes the "previous" y value only if useful data was found from the column.
		}
		actvalprev = actval;
		fillprev=plotx;
	}
		tmp.x = locmap.size().width;
		tmp.y = locmap.size().height;
		polyfill[0][fillcount] = tmp;
		fillcount++;

		tmp.x = 1;
		tmp.y = locmap.size().height;
		polyfill[0][fillcount] = tmp;
		fillcount++;

//This code fills in the area "behind" the found obstacles
		if(fillcount>50)
		{
			drawContours(locmap,polyfill,-1, // draw all contours
			cv::Scalar(0), // in black
			-1); // fill it in
		}

		imwrite("lookmap.jpg", locmap);		//moo local map processing finished


	for( int xloc=1; xloc < locmap.size().width; xloc++){
		for(int yloc=1; yloc < locmap.size().height; yloc++){
			if(locmap.at<uchar>(yloc,xloc) == 0){
			edittrlocmap(yloc + 490, xloc + 180);
			}
		}
	}




	flip(locmap, locmap, 0);
	flip(trilocmap, trilocmap, 0);
	float angle = 0;
	float distance = 0; // distance in cm
	cout << "input angle in degrees (digits only)" << endl;
	cin >> angle;
	cout << "input distance in centimeters (digits only)" << endl;
	cin >> distance;
	imahe = trilocmap;
	areamap = imread("starmap.bmp");
	areamap.convertTo(areamap,CV_8U);	
	cvtColor(areamap, areamap, CV_BGR2GRAY);

	for( int arx=1; arx < areamap.size().width; arx++){
		for( int ary=1; ary < areamap.size().height; ary++){
			if( areamap.at<uchar>(ary,arx) == 150){
				areamap.at<uchar>(ary,arx) = 255;
			}
		}
	}

	rotim = rotate( angle, imahe );
	int quadrant = getquad(angle);
cout << "quadrant = " << quadrant << endl;
	float var;	
	if( angle > 180){
		var = angle - 180;
		angle = 180 - var;
	}
	if( angle == 180){
		angle = 0;
	}
	int pixdx = mtopixdx(distance, angle);
	int pixdy = mtopixdy(distance, angle);

	resize(rotim, rotim, Size(), 0.5,0.5,INTER_NEAREST );
	rotim.convertTo(rotim,CV_8U);	
	//cvtColor(rotim, rotim, CV_BGR2GRAY);
	
	for( int xloc=1; xloc < rotim.size().width; xloc++){
		for(int yloc=1; yloc < rotim.size().height; yloc++){
			if(rotim.at<uchar>(yloc,xloc) > 0){
				int xnew = xloc + xbuff;
				int ynew = yloc + ybuff;
				if(quadrant == 90){
					if(rotim.at<uchar>(yloc,xloc) == 255){
						editarmaploc(ynew, xnew - pixdx);
					}
					if(rotim.at<uchar>(yloc,xloc) > 0 && rotim.at<uchar>(yloc,xloc) < 255){
						editarmap(ynew, xnew - pixdx);
					}
				}
				if(quadrant == 180){
					if(rotim.at<uchar>(yloc,xloc) == 255){
						editarmaploc(ynew + pixdy, xnew);
					}
					if(rotim.at<uchar>(yloc,xloc) > 0 && rotim.at<uchar>(yloc,xloc) < 255){
						editarmap(ynew + pixdy, xnew);
					}
				}
				if(quadrant == 270){
					if(rotim.at<uchar>(yloc,xloc) == 255){
						editarmaploc(ynew, xnew + pixdx);
						
					}
					if(rotim.at<uchar>(yloc,xloc) > 0 && rotim.at<uchar>(yloc,xloc) < 255){
						editarmap(ynew, xnew + pixdx);
					}
				}
				if(quadrant == 0){
					if(rotim.at<uchar>(yloc,xloc) == 255){
						editarmaploc(ynew - pixdy, xnew);
					}
					if(rotim.at<uchar>(yloc,xloc) > 0 && rotim.at<uchar>(yloc,xloc) < 255){
						editarmap(ynew - pixdy, xnew);
					}
				}
				if(quadrant == 2){
					if(rotim.at<uchar>(yloc,xloc) == 255){
						editarmaploc(ynew + pixdy, xnew - pixdx);
					}
					if(rotim.at<uchar>(yloc,xloc) > 0 && rotim.at<uchar>(yloc,xloc) < 255){
					editarmap(ynew + pixdy, xnew - pixdx);
					}
				}
				if(quadrant == 3){
					if(rotim.at<uchar>(yloc,xloc) == 255){
						editarmaploc(ynew + pixdy, xnew + pixdx);
					}
					if(rotim.at<uchar>(yloc,xloc) > 0 && rotim.at<uchar>(yloc,xloc) < 255){
						editarmap(ynew + pixdy, xnew + pixdx);
					}
				}
				if(quadrant == 4){
					if(rotim.at<uchar>(yloc,xloc) == 255){
						editarmaploc(ynew - pixdy, xnew + pixdx);
					}
					if(rotim.at<uchar>(yloc,xloc) > 0 && rotim.at<uchar>(yloc,xloc) < 255){
						editarmap(ynew - pixdy, xnew + pixdx);
					}
				}
				if(quadrant == 1){
					if(rotim.at<uchar>(yloc,xloc) == 255){
						editarmaploc(ynew - pixdy, xnew - pixdx);
					}
					if(rotim.at<uchar>(yloc,xloc) > 0 && rotim.at<uchar>(yloc,xloc) < 255){
						editarmap(ynew - pixdy, xnew - pixdx);
					}
				}


			}
		}
	}
				if(quadrant == 90){
					xbuff= xbuff - pixdx;
				}
				if(quadrant == 180){
					ybuff= ybuff + pixdy;
				}
				if(quadrant == 270){
					xbuff= xbuff + pixdx;
				}
				if(quadrant == 0){
					ybuff= ybuff - pixdy;
				}
				if(quadrant == 2){
					ybuff= ybuff + pixdy;
					xbuff= xbuff - pixdx;
				}
				if(quadrant == 3){
					ybuff= ybuff + pixdy;
					xbuff= xbuff + pixdx;
				}
				if(quadrant == 4){
					ybuff= ybuff - pixdy;
					xbuff= xbuff + pixdx;
				}
				if(quadrant == 1){
					ybuff= ybuff - pixdy;
					xbuff= xbuff - pixdx;
				}

	imwrite("starmap.bmp", areamap);
	imwrite("trilocmapfill.bmp", trilocmap);
	for( int xlocmap=1; xlocmap < locmap.size().width; xlocmap++){
		for(int ylocmap=1; ylocmap < locmap.size().height; ylocmap++){
			locmap.at<uchar>(ylocmap, xlocmap) = 255;
		}
	}
	}

     }
	
	waitKey(0);
	return 0;
}
