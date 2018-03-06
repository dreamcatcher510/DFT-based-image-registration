#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "DemoDFT.h"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char* progName)
{
    cout << endl
        <<  "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        <<  "The dft of an image is taken and it's power spectrum is displayed."          << endl
        <<  "Usage:"                                                                      << endl
        << progName << " [image_name -- default lena.jpg] "                       << endl << endl;
}

//int main(int argc, char ** argv)
//{
////    help(argv[0]);
////
////    const char* filename = argc >=2 ? argv[1] : "lena.jpg";
////
////    Mat I = imread("C:\\opencv\\sources\\samples\\cpp\\lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
////    if( I.empty())
////        return -1;
//
//    /** generate two test images */
//    Mat a(400, 400, CV_8UC1, Scalar(0));
//    Mat b = Mat::zeros(a.size(), a.type());
//
//    /// generate a figure in the image I1
//
//    RotatedRect rRect_a = RotatedRect(Point2f(200,200), Size2f(40,20), 0);
//    Point2f vertices_a[3];
//    rRect_a.points(vertices_a);
//    for (int i = 0; i < 3; i++)
//        line(a, vertices_a[i], vertices_a[(i+1)%3], Scalar(255));
//
//    /// generate a figure in the image I2 by transforming I1
//
//    /// first step: make a translation by changing the position of the center point
//    Point2f rCentre(200, 200);
//    RotatedRect rRect_b = RotatedRect(rCentre, Size2f(40,20), 0);
//    Point2f vertices_b[3];
//    rRect_b.points(vertices_b);
//    for (int i = 0; i < 3; i++)
//        line(b, vertices_b[i], vertices_b[(i+1)%3], Scalar(255));
//    /// second step: make a rotation and scale by computing a rotation matrix
//    /// with respect to the center of the image
//    Point center_transform = Point( b.cols/2, b.rows/2 );
//    double angle_transform = 10;
//    double scale_transform = 1.0;
//    Mat rot_mat_transform( 2, 3, CV_32FC1 );
//    /// Get the rotation matrix with the specifications above
//    rot_mat_transform = getRotationMatrix2D( center_transform, angle_transform, scale_transform );
//    /// transform the image a to get the transformed image b
//    warpAffine( b, b, rot_mat_transform, a.size() );
//
//    /// show the test images
//    if (a.empty() || b.empty())
//        return -1;
//
//    imshow("a", a);
//    imshow("b", b);
//
//    Mat I = b.clone();
//
//    Mat padded;                            //expand input image to optimal size
//    int m = getOptimalDFTSize( I.rows );
//    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
//    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
//
//    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
//    Mat complexI;
//    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
//
//    dft(complexI, complexI);            // this way the result may fit in the source matrix
//
//    // compute the magnitude and switch to logarithmic scale
//    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
//    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
//    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
//    Mat magI = planes[0];
//
//    magI += Scalar::all(1);                    // switch to logarithmic scale
//    log(magI, magI);
//
//    // crop the spectrum, if it has an odd number of rows or columns
//    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
//
//    // rearrange the quadrants of Fourier image  so that the origin is at the image center
//    int cx = magI.cols/2;
//    int cy = magI.rows/2;
//
//    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
//    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
//    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
//    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
//
//    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
//    q0.copyTo(tmp);
//    q3.copyTo(q0);
//    tmp.copyTo(q3);
//
//    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
//    q2.copyTo(q1);
//    tmp.copyTo(q2);
//
//    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
//                                            // viewable image form (float between values 0 and 1).
//
//    imshow("Input Image"       , I   );    // Show the result
//    imshow("spectrum magnitude", magI);
//    waitKey();
//
//    return 0;
//}

int main(int argc, char ** argv)
{
//    help(argv[0]);
//
//    const char* filename = argc >=2 ? argv[1] : "lena.jpg";
//
//    Mat I = imread("C:\\opencv\\sources\\samples\\cpp\\lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    if( I.empty())
//        return -1;

    DemoDFT demo;
    demo.set_values(40);
    demo.demoMagDFTLogPolarPCSimuVideo();


//    FemtoTrack femtoTrack;

    waitKey();

    return 0;
}

