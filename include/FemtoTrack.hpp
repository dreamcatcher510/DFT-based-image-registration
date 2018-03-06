#ifndef FEMTOTRACK_HPP
#define FEMTOTRACK_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv/cxcore.hpp>

#define PI 3.1415926535897932384626433832795

using namespace cv;

class FemtoTrack
{
    public:
        FemtoTrack();
        virtual ~FemtoTrack();
        void ftTrack1(Mat&, Mat&, double&,  double&, double&, double&, double&);  // direct recovery of Tx, Ty, Rz, Scale
        void ftTrack2(Mat&, Mat&, double&, double&, double&, double&, double&, double&);  // with Canny recovery of Tx, Ty, Rz, Scale
        void ftTrack3(Mat&, Mat&, double&,  double&, double&, double&, double&);  // long recovery of Tx, Ty, Rz, Scale
        void focusTrack();   // recovery of Tz
        Mat rotateImg(Mat&, Point2f, double, double);   // input img, centre of rotation, angle or rotation in degrees
        Mat translateImg(Mat&, double, double);   // input img, x translation, y translation in pixels
        Mat magnitudeDFT(Mat&);     // compute the magnitude of the DFT of an image
        void showHistogram(Mat&);  // compute and show histogram of an image
        void skeleton(Mat&, Mat&);  // computethe skeleton of an image
        void showProfile(Mat&, int&); // show the profile of the row
        void copyRowToFile(Mat& img, int& row, const char* file);   //save row into a file

        /// added
        FemtoTrack(int);
        void set_m(int m);
        int  read_m();
        Mat logPolar(Mat&);
        Mat translateImg(Mat& img, Point2f offset);
        void tranRotate(Mat& src,double angle,Point* _org=NULL);
        void tranShift(Mat& src,Point2f& offset);
        Mat CannyThreshold(Mat &src_gray);

        /// rewrite the phaseCorrelate module of OpenCV
        void magSpectrums( InputArray _src, OutputArray _dst);
        void divSpectrums( InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB);
        void fftShift(InputOutputArray _out);
        Point2d weightedCentroid(InputArray _src, cv::Point peakLocation, cv::Size weightBoxSize, double* response);
        Point2d phaseCorrelateRes(InputArray _src1, InputArray _src2, InputArray _window, double* response);
    protected:
        int _m;
    private:
};

inline
double pi_to_pi(double angle) {
	while ( angle < -CV_PI )
		angle += 2.*CV_PI;
	while ( angle > CV_PI )
		angle -= 2.*CV_PI;
	return angle;
}

inline
double angleNormalization(double angle) {
	while ( angle < -180 )
		angle += 360;
	while ( angle > 180 )
		angle -= 360;
	return angle;
}

#endif // FEMTOTRACK_HPP
