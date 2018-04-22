#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QGraphicsScene>
#include "test.cpp"
#include <QDebug>

#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <windows.h>

#include <gl/glew.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <GL/glut.h>

#include <NuiApi.h>
#include <NuiImageCamera.h>
#include <NuiSensor.h>
#include <NuiSkeleton.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <unistd.h>  // check if needed. http://stackoverflow.com/questions/22705751/cannot-open-include-file-unistd-h-no-such-file-or-directory
#include <iostream>
#include <math.h>

#define PI 3.14159265

#define imageWidth 640
#define imageHeight 480
#define CHANNEL 3

using namespace cv;
using namespace std;

BYTE buf[imageWidth * imageHeight * CHANNEL];


// OpenGL Variables
long depthToRgbMap[imageWidth*imageHeight*2];
// We'll be using buffer objects to store the kinect point cloud
GLuint vboId;
GLuint cboId;

// Kinect variables
HANDLE depthStream;
HANDLE rgbStream;
INuiSensor* sensor;

// Mat variables from initial code
Mat image;
Mat imahe;
Mat showThis;
Mat depthMap;
Mat bgrImage;
Mat rotim;
Mat areamap;
Mat rotmat;
Mat trilocmap;
Mat buff1;
Mat locmap(450,640,CV_8U,cv::Scalar(255));  //initialize an all-white 640x450 local map image
cv::Mat image1(1000,1000,CV_8U,cv::Scalar(255));    //initialize an all-white 1000x1000 image
Mat depthmmImage = Mat(imageHeight,imageWidth,CV_8UC1);

std::vector<std::vector<cv::Point> > contours;

// Functions from initial code
Mat rotate(int angle, Mat image);
float mtopixdy(int distance, int angle);
float mtopixdx(int distance, int angle);
int getquad(int angle);
void editarmap(int y, int x);
void editarmaploc(int y, int x);
float pix2m (int pixval);
void editlocmap(int pixval, int x);
void edittrlocmap(int y, int x);

inline QImage  cvMatToQImage( const cv::Mat &inMat )
{
  switch ( inMat.type() )
  {
     // 8-bit, 4 channel
     case CV_8UC4: // RGB Image is this type
     {
        //qDebug() << "[cvMatToQImage] type: CV_8UC4";
        QImage imageVar( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB32 );

        return imageVar;
     }

     // 8-bit, 3 channel
     case CV_8UC3:  // Depth Map is this type
     {
        //qDebug() << "[cvMatToQImage] type: CV_8UC3";
        /* QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB888 );
         *
         * NOTE: CHANGED inMat.step in above to inMat.cols to avoid crashing!!!!
         * Source: http://stackoverflow.com/questions/5516574/qpixmapfromimage-gives-segmentation-fault-in-qx11pixmapdata
         */
        QImage imageVar( inMat.data, inMat.cols, inMat.rows, inMat.cols, QImage::Format_RGB888 );
        // qDebug() << "[cvMatToQImage] type: CV_8UC3: created temp image. Converting to image.rgbSwapped()..";
        imageVar = imageVar.rgbSwapped();
        // qDebug() << "[cvMatToQImage] type: CV_8UC3: RGB Swapped successful. Returning...";
        return imageVar;
     }

     // 8-bit, 1 channel
     case CV_8UC1:
     {
        //qDebug() << "[cvMatToQImage] type: CV_8UC1";
        static QVector<QRgb>  sColorTable;

        // only create our color table once
        if ( sColorTable.isEmpty() )
        {
           for ( int i = 0; i < 256; ++i )
              sColorTable.push_back( qRgb( i, i, i ) );
        }

        QImage imageVar( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_Indexed8 );

        imageVar.setColorTable( sColorTable );

        return imageVar;
     }

     default:
        qWarning() << "ASM::cvMatToQImage() - cv::Mat image type not handled in switch:" << inMat.type();
        break;
  }

  return QImage();
}


/**
 *  Kinect functions: Initialization, etc.
 *
 **/

bool initKinect() {
    // Get a working kinect sensor
    int numSensors;
    if (NuiGetSensorCount(&numSensors) < 0 || numSensors < 1) {
        qDebug() << "Kinect not found. Aborting...";
        return false;
    }
    if (NuiCreateSensorByIndex(0, &sensor) < 0) {
        qDebug() << "NuiCreateSensorByIndex returned negative value. Aborting...";
        return false;
    }

    // Initialize sensor
    sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH | NUI_INITIALIZE_FLAG_USES_COLOR);
    sensor->NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH, // Depth camera or rgb camera?
        NUI_IMAGE_RESOLUTION_640x480,                // Image resolution
        0,        // Image stream flags, e.g. near mode
        2,        // Number of frames to buffer
        NULL,     // Event handle
        &depthStream);
    sensor->NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, // Depth camera or rgb camera?
        NUI_IMAGE_RESOLUTION_640x480,                // Image resolution
        0,      // Image stream flags, e.g. near mode
        2,      // Number of frames to buffer
        NULL,   // Event handle
        &rgbStream);
    return sensor;
}

void getDepthData(GLubyte* dest) {
    float* fdest = (float*) dest;
    long* depth2rgb = (long*) depthToRgbMap;
    NUI_IMAGE_FRAME imageFrame;
    NUI_LOCKED_RECT LockedRect;
    if (sensor->NuiImageStreamGetNextFrame(depthStream, 0, &imageFrame) < 0) {
        qDebug() << "Failed to get next Depth image frame. Aborting...";
        return;
    }
    INuiFrameTexture* texture = imageFrame.pFrameTexture;
    texture->LockRect(0, &LockedRect, NULL, 0);
    if (LockedRect.Pitch != 0) {
        const USHORT* curr = (const USHORT*) LockedRect.pBits;
        for (int j = 0; j < imageHeight; ++j) {
            for (int i = 0; i < imageWidth; ++i) {
                // Get depth of pixel in millimeters
                USHORT depth = NuiDepthPixelToDepth(*curr++);
                // Store coordinates of the point corresponding to this pixel
                Vector4 pos = NuiTransformDepthImageToSkeleton(i, j, depth<<3, NUI_IMAGE_RESOLUTION_640x480);
                *fdest++ = pos.x/pos.w;
                *fdest++ = pos.y/pos.w;
                *fdest++ = pos.z/pos.w;
                // Store the index into the color array corresponding to this pixel
                NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(
                    NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, NULL,
                    i, j, depth<<3, depth2rgb, depth2rgb+1);
                depth2rgb += 2;
            }
        }
    }
    texture->UnlockRect(0);
    sensor->NuiImageStreamReleaseFrame(depthStream, &imageFrame);
}

void getRgbData(GLubyte* dest) {
    float* fdest = (float*) dest;
    long* depth2rgb = (long*) depthToRgbMap;
    NUI_IMAGE_FRAME imageFrame;
    NUI_LOCKED_RECT LockedRect;
    if (sensor->NuiImageStreamGetNextFrame(rgbStream, 0, &imageFrame) < 0) {
        qDebug() << "Failed to get next RGB frame. Aborting...";
        return;
    }
    INuiFrameTexture* texture = imageFrame.pFrameTexture;
    texture->LockRect(0, &LockedRect, NULL, 0);
    if (LockedRect.Pitch != 0) {
        const BYTE* start = (const BYTE*) LockedRect.pBits;
        for (int j = 0; j < imageHeight; ++j) {
            for (int i = 0; i < imageWidth; ++i) {
                // Determine rgb color for each depth pixel
                long x = *depth2rgb++;
                long y = *depth2rgb++;
                // If out of bounds, then don't color it at all
                if (x < 0 || y < 0 || x > imageWidth || y > imageHeight) {
                    for (int n = 0; n < 3; ++n) *(fdest++) = 0.0f;
                }
                else {
                    const BYTE* curr = start + (x + imageWidth*y)*4;
                    for (int n = 0; n < 3; ++n) *(fdest++) = curr[2-n]/255.0f;
                }

            }
        }
    }
    texture->UnlockRect(0);
    sensor->NuiImageStreamReleaseFrame(rgbStream, &imageFrame);
}

void getKinectData() {
    const int dataSize = imageWidth*imageHeight*3*4;
    GLubyte* ptr;
    glBindBuffer(GL_ARRAY_BUFFER, vboId);
    glBufferData(GL_ARRAY_BUFFER, dataSize, 0, GL_DYNAMIC_DRAW);
    ptr = (GLubyte*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (ptr) {
        getDepthData(ptr);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, cboId);
    glBufferData(GL_ARRAY_BUFFER, dataSize, 0, GL_DYNAMIC_DRAW);
    ptr = (GLubyte*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (ptr) {
        getRgbData(ptr);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
}

void rotateCamera() {
    static double angle = 0.;
    static double radius = 3.;
    double x = radius*sin(angle);
    double z = radius*(1-cos(angle)) - radius/2;
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(x,0,z,0,0,radius/2,0,1,0);
    angle += 0.05;
}

void drawKinectData() {
    getKinectData();
    rotateCamera();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, vboId);
    glVertexPointer(3, GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ARRAY_BUFFER, cboId);
    glColorPointer(3, GL_FLOAT, 0, NULL);

    glPointSize(1.f);
    glDrawArrays(GL_POINTS, 0, imageWidth*imageHeight);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

void draw() {
   drawKinectData();
   glutSwapBuffers();
}

void execute() {
    glutMainLoop();
}

bool init() {
    // glutInit is placed in main.cpp
    //glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(imageWidth,imageHeight);
    glutCreateWindow("Point Cloud");
    glutDisplayFunc(draw);
    glutIdleFunc(draw);
    glewInit();
    return true;
}


/**
 * Kinect to OpenCV Utility functions
 *
 **/

inline QPixmap cvMatToQPixmap( const cv::Mat &inMat )
{
    //qDebug() << "[cvMatToQPixmap] Entered cvMatToQPixmap. Sending to cvMatToQImage..";
    QImage temp = cvMatToQImage(inMat);
    //qDebug() << "[cvMatToQPixmap] Converted to QImage. Converting to Pixmap..";
    QPixmap tempPixmap = QPixmap::fromImage(temp);
    //qDebug() << "[cvMatToQPixmap] converted to Pixmap. Returning...";
    return tempPixmap;
    //return QPixmap::fromImage( cvMatToQImage( inMat ) );
}

cv::Mat getRgbImage() {
    cv::Mat rgbMat;
    IplImage* color = cvCreateImageHeader(cvSize(imageWidth, imageHeight), IPL_DEPTH_8U, 4);
    const NUI_IMAGE_FRAME * pImageFrame = NULL;
    HRESULT hr = NuiImageStreamGetNextFrame(
        rgbStream,
        250, //no. of ms to wait until next frame
        &pImageFrame);
    if (FAILED(hr))
    {
        qDebug() << "Get RGB Image Frame Failed";
    }
    //NuiImageBuffer * pTexture = pImageFrame->pFrameTexture;
    INuiFrameTexture* pTexture = pImageFrame->pFrameTexture;
    NUI_LOCKED_RECT LockedRect;
    pTexture->LockRect(0, &LockedRect, NULL, 0);
    if (LockedRect.Pitch != 0)
    {
        qDebug() << "Getting RGB Image...";
        BYTE* pBuffer = (BYTE*) LockedRect.pBits;
        cvSetData(color, pBuffer, LockedRect.Pitch);
        rgbMat = cv::Mat(color);
        qDebug() << "Got RGB Image";
    } else {
        qDebug() << "Failed to get RGB Image";
    }
    pTexture->UnlockRect(0);
    NuiImageStreamReleaseFrame(rgbStream, pImageFrame);
    return rgbMat;
}

cv::Mat getDepthImage() {
    cv::Mat depthMat;
    IplImage* depth = cvCreateImageHeader(cvSize(imageWidth, imageHeight), IPL_DEPTH_8U, 3);
    const NUI_IMAGE_FRAME * pImageFrame = NULL;
    HRESULT hr = NuiImageStreamGetNextFrame(
        depthStream,
        700, //no. of ms to wait until next frame
        &pImageFrame);
    if (FAILED(hr))
    {
        qDebug() << "Get depth Image Frame Failed";
    }
    //NuiImageBuffer * pTexture = pImageFrame->pFrameTexture;
    INuiFrameTexture* pTexture = pImageFrame->pFrameTexture;
    NUI_LOCKED_RECT LockedRect;
    pTexture->LockRect(0, &LockedRect, NULL, 0);
    if (LockedRect.Pitch != 0)
    {
        qDebug() << "Getting depth image...";
        USHORT * pBuff = (USHORT*) LockedRect.pBits;
        for (int i = 0; i < imageWidth * imageHeight; i++)
        {
            BYTE index = pBuff[i] & 0x07;
            USHORT realDepth = (pBuff[i] & 0xFFF8) >> 3;
            BYTE scale = 255 - (BYTE)(256 * realDepth / 0x0fff);
            buf[CHANNEL * i] = buf[CHANNEL * i + 1] = buf[CHANNEL * i + 2] = 0;
            switch (index)
            {
            case 0:
                buf[CHANNEL * i] = scale / 2;
                buf[CHANNEL * i + 1] = scale / 2;
                buf[CHANNEL * i + 2] = scale / 2;
                break;
            case 1:
                buf[CHANNEL * i] = scale;
                break;
            case 2:
                buf[CHANNEL * i + 1] = scale;
                break;
            case 3:
                buf[CHANNEL * i + 2] = scale;
                break;
            case 4:
                buf[CHANNEL * i] = scale;
                buf[CHANNEL * i + 1] = scale;
                break;
            case 5:
                buf[CHANNEL * i] = scale;
                buf[CHANNEL * i + 2] = scale;
                break;
            case 6:
                buf[CHANNEL * i + 1] = scale;
                buf[CHANNEL * i + 2] = scale;
                break;
            case 7:
                buf[CHANNEL * i] = 255 - scale / 2;
                buf[CHANNEL * i + 1] = 255 - scale / 2;
                buf[CHANNEL * i + 2] = 255 - scale / 2;
                break;
            }
        }


        cvSetData(depth, buf, imageWidth * CHANNEL);
        //cvNamedWindow("depth image", CV_WINDOW_AUTOSIZE);
        //cvShowImage("depth image", depth);
        depthMat = cv::Mat(depth);


        const USHORT* curr = (const USHORT*) LockedRect.pBits;
        const USHORT* dataEnd = curr + (imageWidth*imageHeight);

        int depthX = 0;
        int depthY = 0;

        while (curr < dataEnd) {
            // Get depth in millimeters
            USHORT depthmm = NuiDepthPixelToDepth(*curr++);
            BYTE depthByte = (BYTE) depthmm%256;

            if (depthX < imageWidth && depthY < imageHeight) {
                depthmmImage.at<uchar>(depthY,depthX) = depthByte;
            } else if (depthX >= imageWidth) {
                depthX = 0;
                depthY++;
                depthmmImage.at<uchar>(depthY,depthX) = depthByte;
            }
            depthX++;

            //qDebug() << "depthMM remainder 256: " << depthByte;

            /*// Draw a grayscale image of the depth:
            // B,G,R are all set to depth%256, alpha set to 1.
            for (int i = 0; i < 3; ++i) {
                //qDebug() << "depthmm: " << depthmm;
                //*dest++ = (BYTE) depthmm%256;
            //*dest++ = 0xff;
            }*/

        }
        cv::flip(depthmmImage, depthmmImage, 1);                    // Flip to the correct orientation
        imshow("DEPTH MAP", depthmmImage);
        qDebug() << "Got depth Image";
    } else {
        qDebug() << "Failed to get depth image";
    }
    //pTexture->UnlockRect(0);
    NuiImageStreamReleaseFrame(depthStream, pImageFrame);
    return depthMat;
}



/**
 * UI-related functions
 *
 **/


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{

    if (!initKinect()) {
        qDebug() << "[MainWindow::MainWindow] Failed to initialize Kinect.";
    } else {
        qDebug() << "[MainWindow::MainWindow] Kinect successfully initialized";
    }

    // OpenGL setup
    glClearColor(0,0,0,0);
    glClearDepth(1.0f);


    ui->setupUi(this);

    /*
     *  Show RGB Image in the top left + scale
     */
    cv::Mat inputImage = getRgbImage();
    QPixmap rgbPixmap = cvMatToQPixmap(inputImage);

    // Flip the rgbImage for the user
    QImage tempQImage = rgbPixmap.toImage();
    tempQImage = tempQImage.mirrored(true, false);
    rgbPixmap = QPixmap::fromImage(tempQImage);

    qGraphicsScene = new QGraphicsScene(this);
    qGraphicsScene->addPixmap(rgbPixmap);
    qGraphicsScene->setSceneRect(rgbPixmap.rect());
    ui->rgbImage->setScene(qGraphicsScene);
    ui->rgbImage->scale(0.5,0.5);                       // Only scale the first time showing the image

    /*if (rgbPixmap.save("C:/Users/MOOMOO/Desktop/gui/WRIIITE.jpg", 0, 100)) {
        qDebug() << "Image save successful!";
    } else {
        qDebug() << "Image save failed.";
    }*/

    // Dummy images
    qPixmapImage.load("C:/Users/MOOMOO/Desktop/gui/rei.png");
    qGraphicsScene = new QGraphicsScene(this);
    qGraphicsScene->addPixmap(qPixmapImage);
    qGraphicsScene->setSceneRect(qPixmapImage.rect());
    ui->pointCloud->setScene(qGraphicsScene);
    ui->areaMap->setScene(qGraphicsScene);


    /*
     *  Get depth image
     */
            cv::Mat depthMat = getDepthImage();                 // At this point, depthMat is still BGR
            //const float scaleFactor = 0.06f;                  // change scaling accordingly
            const float scaleFactor = 1;                        // change scaling accordingly
            depthMat.convertTo( depthMat, CV_8UC1, scaleFactor);// Convert depth map to needed format
            cv::flip(depthMat, depthMat, 1);                    // Flip to the correct orientation

            /*
             *  Process depth image
             */
            image = depthMat;
            image = image.rowRange(1,475);                      // CROP the image to avoid depth map noise
            imwrite("C:/Users/MOOMOO/Desktop/gui/images/depthmap.bmp", depthMat);

            Mat trilocmapTemp = imread("C:/Users/MOOMOO/Desktop/gui/angle.bmp", 1);
            trilocmapTemp.convertTo(trilocmapTemp,CV_8U);
            cvtColor(trilocmapTemp, trilocmap, CV_BGR2GRAY);

            Mat logsobel;
            Sobel(image,logsobel,CV_8U,0,1,5,80,0);             //vertical sobel filter: changed last parameter to 0
            medianBlur(logsobel,logsobel,9);

            logsobel.convertTo(logsobel, CV_64F, 1);            //convert image; done for logarithm filter
            log(logsobel, logsobel);

            logsobel.convertTo(logsobel,CV_8U, 255, 255);       //convert the image to 8-bit grayscale for findContours() function
            bitwise_not(logsobel, logsobel);                     // INVERT BW

            //logsobel.convertTo(logsobel, CV_8UC1, 1);           //need to convert back for dilation and erosion
            dilate(logsobel,logsobel,Mat(),Point(-1,-1), 3);    //dilate and erode to remove noise
            erode(logsobel,logsobel,Mat(),Point(-1,-1), 1);     //change last param to 7?????
            //logsobel.convertTo(logsobel, CV_64F, 1);            //convert back to CV_64F?

            /*
             * For debugging purposes
             */
            /*for (int x = 0; x < logsobel.size().width ; x = x + 20) {
                int y=441;
                //for (int y = 454; y < 459; y++) {
                    qDebug() << "(" << x << "," << y << ") = " << logsobel.at<uchar>(y, x);
                //}
            }
            imwrite("C:/Users/MOOMOO/Desktop/gui/images/debug.png", logsobel);*/

            //Prepare the image for findContours
            cv::threshold(logsobel, logsobel, 50, 255, CV_THRESH_BINARY);
            //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
            //std::vector<std::vector<cv::Point> > contours;
            //cv::Mat contourOutput = logsobel.clone();
            //cv::findContours( contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
            Mat logsobelGray;
            cvtColor(logsobel, logsobelGray, CV_BGR2GRAY);

            /*Mat image5;
            image5 = imread("C:/Users/MOOMOO/Desktop/gui/shape.jpg", 1);  
            Mat gray;
            cvtColor(image5, gray, CV_BGR2GRAY);
            Canny(gray, gray, 100, 200, 3);
            /// Find contours   
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            RNG rng(12345);
            //findContours( gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            */


            //Parameters for cvFindContours
            static const int thickness = 5;//CV_FILLED - filled contour
            static const int lineType = 8;//8:8-connected,  4:4-connected line, CV_AA: anti-aliased line.
            Scalar           color = CV_RGB(255, 20, 20); // line color - light red
            Scalar           hole_color = CV_RGB(20,255,20);

            //Segmented image
            Mat SegmentedVar = logsobelGray > 128;

            /// Find contours - use old style C since C++ version has bugs. 

            if (SegmentedVar.depth() == CV_8U)
                  qDebug() << "logsobel = Unsigned char image" << endl;
            qDebug() << "logsobel: Number of channels is: " << SegmentedVar.channels() << endl;

            Mat             drawingVar(logsobelGray.size(), CV_8U,
                cv::Scalar(255));
            //cvtColor(logsobelGray, drawingVar, CV_GRAY2RGB);
            IplImage        drawingIplVar = drawingVar;

            IplImage        SegmentedIplVar = SegmentedVar;
            CvMemStorage*   storageVar = cvCreateMemStorage(0);
            CvSeq*          contoursVar = 0;
            int             numContVar = 0;
            int             contAthreshVar = 45;

            numContVar = cvFindContours(&SegmentedIplVar, storageVar, &contoursVar, sizeof(CvContour),
                CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0,0));

            //qDebug() << "numContVar = " << numContVar;



            /*CvSeq *it = contoursVar;
            double smallArea = (20*475);
            while (it)
            {
                double a = cvContourArea(it, CV_WHOLE_SEQ, false);
                if (a < smallArea)
                {
                    //smallArea = a;
                    //qDebug() << "smallest area = " << a;
                    //largest_contour_index = i;
                    //bounding_rect = boundingRect(it);
                    cvSeqPop(it);
                }
                it = it->h_next;
            }*/


            for (; contoursVar != 0; contoursVar = contoursVar->h_next)
            {
                //cvDrawContours(&drawingIplVar, contoursVar, color, 
                 //   hole_color, -1, -1, lineType, cvPoint(0, 0));
                cvDrawContours(&drawingIplVar, contoursVar,
                    cv::Scalar(1),  // black
                    cv::Scalar(1),  // black
                    -1,             // draw all contours
                    -1,             // fill it in
                    lineType,
                    cvPoint(0,0));
            }

            // Use opening technique (erode+dilate) to remove "noisy" contours
            int erosion_size = 10; // adjust with you application
            Mat erode_element = getStructuringElement( MORPH_ELLIPSE,
                                     Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                     Point( erosion_size, erosion_size ) );
            erode( drawingVar, drawingVar, erode_element );
            dilate( drawingVar, drawingVar, erode_element );


            //Mat drawing = cvarrToMat(&drawingIpl); //Mat(&IplImage) is soon to be deprecated OpenCV 3.X.X
            imshow("drawingVar", drawingVar);
            qDebug() << "drawingVar.size() = " << drawingVar.size().width << ","<< drawingVar.size().height;

            image = depthmmImage.rowRange(1,475);   //TODO remove this
            Mat dest = image.clone();
            //cvtColor( image, dest, CV_RGB2GRAY ); //TODO return this
            image = dest - drawingVar;
            imshow("image", image);


            // to get actual mm values from image, simply reverse modulo 256 each element


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

                if (pixprev>=255 && actprev!=0 && consec<5)     //if no data was useful from the processed column
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
                    actprev=actval;     //changes the "previous" y value only if useful data was found from the column.
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

            imwrite("C:/Users/MOOMOO/Desktop/gui/images/lookmap.jpg", locmap);     //moo local map processing finished

            for( int xloc=1; xloc < locmap.size().width; xloc++){
                for(int yloc=1; yloc < locmap.size().height; yloc++){
                    if(locmap.at<uchar>(yloc,xloc) == 0){
                    edittrlocmap(yloc + 490, xloc + 180);
                    }
                }
            }


            //TODO check if flipping is necessary?
            flip(locmap, locmap, 0);
            flip(trilocmap, trilocmap, 0);
            imshow("trilocmap",trilocmap);

}

MainWindow::~MainWindow()
{
    delete ui;
    delete qGraphicsScene;
}

void MainWindow::on_pushButton_clicked()
{
    // Convert from matrix to pixmap
    // IplImage* img = cvLoadImage("c://Users/MOOMOO/Desktop/gui/test.png");
    // cv::Mat inputImage = cv::Mat(img);
    // QPixmap pixmap = QPixmap::fromImage(QImage((unsigned char*) inputImage.data, inputImage.cols, inputImage.rows, QImage::Format_RGB888).rgbSwapped());

 /*-------- End OpenCV Testing -----------*/

    // Get x and y coordinates
    QString xString = ui->xCoord->displayText();
    QString yString = ui->yCoord->displayText();

    float angle = xString.toFloat();
    float distance = yString.toFloat(); // distance in cm

    qDebug() << "angle = " << angle;
    qDebug() << "distance = " << distance;


    //////////////////////////////////////////////////////////////////
    //================================================================
    // COPIED CODE FROM MAINWINDOW FUNCTION
    //================================================================
    //////////////////////////////////////////////////////////////////

                /*
             *  Get depth image
             */
            cv::Mat depthMat = getDepthImage();                 // At this point, depthMat is still BGR
            //const float scaleFactor = 0.06f;                  // change scaling accordingly
            const float scaleFactor = 1;                        // change scaling accordingly
            depthMat.convertTo( depthMat, CV_8UC1, scaleFactor);// Convert depth map to needed format
            cv::flip(depthMat, depthMat, 1);                    // Flip to the correct orientation

            /*
             *  Process depth image
             */
            image = depthMat;
            image = image.rowRange(1,475);                      // CROP the image to avoid depth map noise
            imwrite("C:/Users/MOOMOO/Desktop/gui/images/depthmap.bmp", depthMat);

            Mat trilocmapTemp = imread("C:/Users/MOOMOO/Desktop/gui/angle.bmp", 1);
            trilocmapTemp.convertTo(trilocmapTemp,CV_8U);
            cvtColor(trilocmapTemp, trilocmap, CV_BGR2GRAY);

            Mat logsobel;
            Sobel(image,logsobel,CV_8U,0,1,5,80,0);             //vertical sobel filter: changed last parameter to 0
            medianBlur(logsobel,logsobel,9);

            logsobel.convertTo(logsobel, CV_64F, 1);            //convert image; done for logarithm filter
            log(logsobel, logsobel);

            logsobel.convertTo(logsobel,CV_8U, 255, 255);       //convert the image to 8-bit grayscale for findContours() function
            bitwise_not(logsobel, logsobel);                     // INVERT BW

            //logsobel.convertTo(logsobel, CV_8UC1, 1);           //need to convert back for dilation and erosion
            dilate(logsobel,logsobel,Mat(),Point(-1,-1), 3);    //dilate and erode to remove noise
            erode(logsobel,logsobel,Mat(),Point(-1,-1), 1);     //change last param to 7?????
            //logsobel.convertTo(logsobel, CV_64F, 1);            //convert back to CV_64F?

            /*
             * For debugging purposes
             */
            /*for (int x = 0; x < logsobel.size().width ; x = x + 20) {
                int y=441;
                //for (int y = 454; y < 459; y++) {
                    qDebug() << "(" << x << "," << y << ") = " << logsobel.at<uchar>(y, x);
                //}
            }
            imwrite("C:/Users/MOOMOO/Desktop/gui/images/debug.png", logsobel);*/

            //Prepare the image for findContours
            cv::threshold(logsobel, logsobel, 50, 255, CV_THRESH_BINARY);
            //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
            //std::vector<std::vector<cv::Point> > contours;
            //cv::Mat contourOutput = logsobel.clone();
            //cv::findContours( contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
            Mat logsobelGray;
            cvtColor(logsobel, logsobelGray, CV_BGR2GRAY);

            /*Mat image5;
            image5 = imread("C:/Users/MOOMOO/Desktop/gui/shape.jpg", 1);  
            Mat gray;
            cvtColor(image5, gray, CV_BGR2GRAY);
            Canny(gray, gray, 100, 200, 3);
            /// Find contours   
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            RNG rng(12345);
            //findContours( gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            */


            //Parameters for cvFindContours
            static const int thickness = 5;//CV_FILLED - filled contour
            static const int lineType = 8;//8:8-connected,  4:4-connected line, CV_AA: anti-aliased line.
            Scalar           color = CV_RGB(255, 20, 20); // line color - light red
            Scalar           hole_color = CV_RGB(20,255,20);

            //Segmented image
            Mat SegmentedVar = logsobelGray > 128;

            /// Find contours - use old style C since C++ version has bugs. 

            if (SegmentedVar.depth() == CV_8U)
                  qDebug() << "logsobel = Unsigned char image" << endl;
            qDebug() << "logsobel: Number of channels is: " << SegmentedVar.channels() << endl;

            Mat             drawingVar(logsobelGray.size(), CV_8U,
                cv::Scalar(255));
            //cvtColor(logsobelGray, drawingVar, CV_GRAY2RGB);
            IplImage        drawingIplVar = drawingVar;

            IplImage        SegmentedIplVar = SegmentedVar;
            CvMemStorage*   storageVar = cvCreateMemStorage(0);
            CvSeq*          contoursVar = 0;
            int             numContVar = 0;
            int             contAthreshVar = 45;

            numContVar = cvFindContours(&SegmentedIplVar, storageVar, &contoursVar, sizeof(CvContour),
                CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0,0));

            qDebug() << "numContVar = " << numContVar;



            /*CvSeq *it = contoursVar;
            double smallArea = (20*475);
            while (it)
            {
                double a = cvContourArea(it, CV_WHOLE_SEQ, false);
                if (a < smallArea)
                {
                    //smallArea = a;
                    //qDebug() << "smallest area = " << a;
                    //largest_contour_index = i;
                    //bounding_rect = boundingRect(it);
                    cvSeqPop(it);
                }
                it = it->h_next;
            }*/


            for (; contoursVar != 0; contoursVar = contoursVar->h_next)
            {
                //cvDrawContours(&drawingIplVar, contoursVar, color, 
                 //   hole_color, -1, -1, lineType, cvPoint(0, 0));
                cvDrawContours(&drawingIplVar, contoursVar,
                    cv::Scalar(1),  // black
                    cv::Scalar(1),  // black
                    -1,             // draw all contours
                    -1,             // fill it in
                    lineType,
                    cvPoint(0,0));
            }

            // Use opening technique (erode+dilate) to remove "noisy" contours
            int erosion_size = 10; // adjust with you application
            Mat erode_element = getStructuringElement( MORPH_ELLIPSE,
                                     Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                     Point( erosion_size, erosion_size ) );
            erode( drawingVar, drawingVar, erode_element );
            dilate( drawingVar, drawingVar, erode_element );


            //Mat drawing = cvarrToMat(&drawingIpl); //Mat(&IplImage) is soon to be deprecated OpenCV 3.X.X
            imshow("drawingVar", drawingVar);
            qDebug() << "drawingVar.size() = " << drawingVar.size().width << ","<< drawingVar.size().height;

            Mat dest = image.clone();
            cvtColor( image, dest, CV_RGB2GRAY );
            image = dest - drawingVar;
            imshow("image", image);

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

                if (pixprev>=255 && actprev!=0 && consec<5)     //if no data was useful from the processed column
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
                    actprev=actval;     //changes the "previous" y value only if useful data was found from the column.
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

            imwrite("C:/Users/MOOMOO/Desktop/gui/images/lookmap.jpg", locmap);     //moo local map processing finished

            for( int xloc=1; xloc < locmap.size().width; xloc++){
                for(int yloc=1; yloc < locmap.size().height; yloc++){
                    if(locmap.at<uchar>(yloc,xloc) == 0){
                    edittrlocmap(yloc + 490, xloc + 180);
                    }
                }
            }

            flip(locmap, locmap, 0);
            flip(trilocmap, trilocmap, 0);

    //////////////////////////////////////////////////////////////////
    //================================================================
    // END COPIED CODE FROM MAINWINDOW FUNCTION
    //================================================================
    //////////////////////////////////////////////////////////////////


    imahe = trilocmap;
    //Mat destAreaMap;
    areamap = imread("starmap.bmp");
    areamap.convertTo(areamap,CV_8U);

    if (areamap.depth() == CV_8U)
        qDebug() << "destAreaMap = Unsigned char image";
    qDebug() << "destAreaMap: Number of channels is: " << areamap.channels();

    //cvtColor(destAreaMap, areamap, CV_BGR2GRAY);

    for( int arx=1; arx < areamap.size().width; arx++){
        for( int ary=1; ary < areamap.size().height; ary++){
            if( areamap.at<uchar>(ary,arx) == 150){
                areamap.at<uchar>(ary,arx) = 255;
            }
        }
    }

    rotim = rotate( angle, imahe );
    int quadrant = getquad(angle);
    qDebug() << "quadrant = " << quadrant;
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

    //resize(rotim, rotim, Size(), 0.5,0.5,INTER_NEAREST );
    cv::resize(rotim, rotim, Size(), 0.5,0.5,INTER_NEAREST );
    rotim.convertTo(rotim,CV_8U);   
    //cvtColor(rotim, rotim, CV_BGR2GRAY);
    
    int xbuff= 250;
    int ybuff= 250;
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

////////////// TODO fix crashes in if(quadrant == 0)//////////////////
                /*
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

*/
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

    imwrite("C:/Users/MOOMOO/Desktop/gui/images/starmap.bmp", areamap); // moo show this
    imwrite("C:/Users/MOOMOO/Desktop/gui/images/trilocmapfill.bmp", trilocmap); // moo show this
    for( int xlocmap=1; xlocmap < locmap.size().width; xlocmap++){
        for(int ylocmap=1; ylocmap < locmap.size().height; ylocmap++){
            locmap.at<uchar>(ylocmap, xlocmap) = 255;
        }
    }
    

/*
    qPixmapImage.load("C:/Users/MOOMOO/Desktop/gui/test.png");
    Mat image2;
    image2 = imread( "C:/Users/MOOMOO/Desktop/gui/test.png", 1 );
    imwrite( "C:/Users/MOOMOO/Desktop/gui/WRITE.jpg", image2 );*/


    /*
     *  Update RGB Image in the top left
     */
    cv::Mat rgbImage = getRgbImage();
    QPixmap rgbPixmap = cvMatToQPixmap(rgbImage);

    // Flip the rgbImage for the user
    QImage tempQImage = rgbPixmap.toImage();
    tempQImage = tempQImage.mirrored(true, false);
    rgbPixmap = QPixmap::fromImage(tempQImage);

    qGraphicsScene = new QGraphicsScene(this);
    qGraphicsScene->addPixmap(rgbPixmap);
    qGraphicsScene->setSceneRect(rgbPixmap.rect());
    ui->rgbImage->setScene(qGraphicsScene);


    // Top right
    qPixmapImage.load("C:/Users/MOOMOO/Desktop/gui/test.png");
    qGraphicsScene = new QGraphicsScene(this);
    qGraphicsScene->addPixmap(qPixmapImage);
    qGraphicsScene->setSceneRect(qPixmapImage.rect());
    ui->areaMap->setScene(qGraphicsScene);
    ui->pointCloud->setScene(qGraphicsScene);

    qDebug() << "x = " + xString + "; y = " + yString + ";";
}

void MainWindow::on_pointCloudButton_clicked()
{
    qDebug() << "Point cloud button clicked!";

    if (!init()) {
        qDebug() << "Unable to initialize point cloud window";
    } else {
        qDebug() << "Point cloud window initialized";
    }

    // OpenGL setup
        glClearColor(0,0,0,0);
        glClearDepth(1.0f);

        // Set up array buffers
        glGenBuffers(1, &vboId);
        glBindBuffer(GL_ARRAY_BUFFER, vboId);
        glGenBuffers(1, &cboId);
        glBindBuffer(GL_ARRAY_BUFFER, cboId);

        // Camera setup
        glViewport(0, 0, imageWidth, imageHeight);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, imageWidth /(GLdouble) imageHeight, 0.1, 1000);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(0,0,0,0,0,1,0,1,0);

        // Main loop
        execute();


}


























// Functions from initial code

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
