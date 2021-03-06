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

using namespace cv;
using namespace std;

Mat image;
Mat imahe;
Mat show;
Mat depthMap;
Mat rotim;
Mat areamap;
Mat rotmat;
Mat trilocmap;
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
	float pixdy = (distance*(abs(sin(angle)))/5); //5 cm per pixel
	cout << sin(angle) << endl;
	return pixdy;
}

float mtopixdx(int distance, int angle){
	float pixdx = (distance*(abs(cos(angle)))/5); //5 cm per pixel
	cout << sin(angle) << endl;
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


void marknear(int nearpix){
	int m = pix2m(nearpix);
	for( int xit=1; xit<=locmap.size().width; xit++){
		for(int yit=1; yit<=locmap.size().height; yit++){
			locmap.at<uchar>(yit,xit) = m;
		}
	}	
}

int getnearest()	//returns the nearest pixel value in the entire image
{
	int nearestpixval=255;
	for(int x=1; x<=image.size().width; x++){
		for(int y=1; y<=image.size().height; y++){
			if(((image.at<uchar>(y,x))<nearestpixval) && (image.at<uchar>(y,x)!=0))
				nearestpixval = image.at<uchar>(y,x);
		}
	}
	cout<<"nearest pixval:"<<nearestpixval<<endl;	
	return nearestpixval;
}

int getnearestx(int xit)	//returns the nearest pixel value in a particular column
{
	int nearxpixval=255;
	for(int y=1; y<=image.size().height; y++){
		if(((image.at<uchar>(y,xit))<nearxpixval)&&(image.at<uchar>(y,xit)!=0))
			nearxpixval = image.at<uchar>(y,xit);
	}
	return nearxpixval;
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

	int xbuff= 400;
	int ybuff= 400;
for(int x=0;x<1000;x++)
    {
	char dum;
	cout << "continue? (y/n)" << endl;
	cin >> dum;
	if(dum == 'n'){
		return -1;
	}
	
	capture >> depthMap;

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
		depthMap.convertTo( show, CV_8UC1, scaleFactor );
            }
	image = show;
	
imwrite("depthmap.bmp", image);
	image = image.rowRange(1,475);	//CROP the image to avoid depth map noise	
	trilocmap = imread("angle.bmp");	
	trilocmap.convertTo(trilocmap,CV_8U);	
	cvtColor(trilocmap, trilocmap, CV_BGR2GRAY);
	Mat logsobel;
	
	Sobel(image,logsobel,CV_8U,0,1,3,80,128); //vertical sobel filter
	medianBlur(logsobel,logsobel,5);

	logsobel.convertTo(logsobel, CV_64F, 1);	//convert image; done for logarithm filter
	log(logsobel, logsobel);

	dilate(logsobel,logsobel,Mat(),Point(-1,-1), 3); //dilate and erode to remove noise
	erode(logsobel,logsobel,Mat(),Point(-1,-1), 5);
		
	logsobel.convertTo(logsobel,CV_8U, 255, 255); //convert the image to 8-bit grayscale for findContours() function


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

	int nearpixval=0;
	if (contours.size()==0)
	{
		nearpixval=getnearest();
		marknear(nearpixval);
		return -1;
	}


	int pixval = 0;
	int xcomponent = 0;
	for(int xcomp=65; xcomp < 600; xcomp++){
		for(int ycomp=1; ycomp < result.size().height; ycomp++){
			if (result.at<uchar>(ycomp,xcomp) < 255){
			pixval= pix2m(image.at<uchar>(ycomp, xcomp));
			xcomponent = float(xcomp - (640/2))* (pixval+(-10)) * 0.0021 * (640/480);
			xcomponent = (xcomponent)+320;	
			pixval = pixval;
			editlocmap(pixval, xcomponent);
			}
		}
	}

	int locval= 0;

	for( int xloc=1; xloc < locmap.size().width; xloc++){
		for(int yloc=1; yloc < locmap.size().height; yloc++){
			if(locmap.at<uchar>(yloc,xloc)==0){
			locval=1;
			}
			if(locval==1){
			locmap.at<uchar>(yloc,xloc)=0;
			}
		}
		locval=0;
	}


	for( int xloc=1; xloc < locmap.size().width; xloc++){
		for(int yloc=1; yloc < locmap.size().height; yloc++){
			if(locmap.at<uchar>(yloc,xloc) == 0){
			edittrlocmap(yloc + 380, xloc + 180);
			}
		}
	}




	flip(locmap, locmap, 0);
	flip(trilocmap, trilocmap, 0);

	int angle = 0;
	int distance = 0; // distance in cm
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
				areamap.at<uchar>(ary,arx)=0;
			}
		}
	}

	rotim = rotate( angle, imahe );
	int quadrant = getquad(angle);
	int pixdx = mtopixdx(distance, angle);
	int pixdy = mtopixdy(distance, angle);

	resize(rotim, rotim, Size(), 0.2,0.2,INTER_NEAREST );
	rotim.convertTo(rotim,CV_8U);	
	
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
