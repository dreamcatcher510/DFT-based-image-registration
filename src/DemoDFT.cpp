#include "DemoDFT.h"
#include "FemtoTrack.hpp"

using namespace std;
using namespace cv;

DemoDFT::DemoDFT()
{
    _m = 40;
}

DemoDFT::DemoDFT(int m)
{
    _m = m;
}

DemoDFT::~DemoDFT()
{

}

void DemoDFT::set_values(int m)
{
    _m = m;
}

int DemoDFT::read_values()
{
    return _m;
}

int DemoDFT::demoMagDFT()
{
    /** generate a test image */
    Mat I1(400, 400, CV_8UC1, Scalar(0));
    Mat I2 = Mat::zeros(I1.size(), I1.type());

    /// generate a figure in the image
    RotatedRect rRect_I1 = RotatedRect(Point2f(I1.cols/2,I1.rows/2), Size2f(40,20), 0);
    Point2f vertices_I1[3];
    rRect_I1.points(vertices_I1);
    for (int i = 0; i < 3; i++)
        line(I1, vertices_I1[i], vertices_I1[(i+1)%3], Scalar(255));

    /// generate the second image by transformation
    Point center_transform = Point( I2.cols/2, I2.rows/2 );
    double angle_transform = 30;
    double scale_transform = 1.0;
    Mat rot_mat_transform( 2, 3, CV_32FC1 );
    /// Get the rotation matrix with the specifications above
    rot_mat_transform = getRotationMatrix2D( center_transform, angle_transform, scale_transform );
    /// transform the image a to get the transformed image b
    warpAffine( I1, I2, rot_mat_transform, I1.size() );

    /// show the test images
    if (I1.empty() || I2.empty())
            return -1;

    /** magnitude of DFT */
    FemtoTrack femtoTrack;
    Mat magI2 = femtoTrack.magnitudeDFT(I2);
    normalize(magI2, magI2, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    /** show the results */
    imshow("Input Image"       , I2   );    // Show the result
    imshow("spectrum magnitude", magI2);

    return 0;
}

int DemoDFT::demoCenRotMagDFT()
{
    /** generate a reference image */
    Mat a(200, 200, CV_8UC1, Scalar(0));

    /// generate a figure in the image I1
    RotatedRect rRect_a = RotatedRect(Point2f(a.cols/2,a.rows/2), Size2f(40,20), 0);
    Point2f vertices_a[3];
    rRect_a.points(vertices_a);
    for (int i = 0; i < 3; i++)
        line(a, vertices_a[i], vertices_a[(i+1)%3], Scalar(255));

    /// generate the second image and do magnitude of DFT
    Mat b = Mat::zeros(a.size(), a.type());
    Point center_transform = Point( b.cols/2, b.rows/2 );
    for(int i = 1; i <=360; i++)
    {
        double angle_transform = i;
        double scale_transform = 1.0;
        Mat rot_mat_transform( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform = getRotationMatrix2D( center_transform, angle_transform, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a, b, rot_mat_transform, a.size() );

        /// show the test images
        if (a.empty() || b.empty())
            return -1;

        /** magnitude of DFT */
        FemtoTrack femtoTrack;
        Mat magI = femtoTrack.magnitudeDFT(b);
        normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                                // viewable image form (float between values 0 and 1).

        imshow("Input Image"       , b   );    // Show the result
        imshow("spectrum magnitude", magI);
         //waitKey();
        waitKey(100);
    }
    return 0;
}

int DemoDFT::demoRotMagDFT()
{
    /** generate a reference image */
    /// generate a figure in the image in the center
    Mat a(200, 200, CV_8UC1, Scalar(0));
    Point2f ICenter(a.cols/2,a.rows/2);
    RotatedRect rRect_a = RotatedRect(ICenter, Size2f(40,20), 0);
    Point2f vertices_a[3];
    rRect_a.points(vertices_a);
    for (int i = 0; i < 3; i++)
        line(a, vertices_a[i], vertices_a[(i+1)%3], Scalar(255));

    /// generate a figure in the image with a translation
    Mat a1(200, 200, CV_8UC1, Scalar(0));
    Point2f rCentre_a1 = ICenter + Point2f(60,-60);  // object center
    RotatedRect rRect_a1 = RotatedRect(rCentre_a1, Size2f(40,20), -90);
    Point2f vertices_a1[3];
    rRect_a1.points(vertices_a1);
    for (int i = 0; i < 3; i++)
        line(a1, vertices_a1[i], vertices_a1[(i+1)%3], Scalar(255));

    /// generate the rotated image and do magnitude of DFT
    Mat b = Mat::zeros(a.size(), a.type()); // rotate with
    Mat b1 = Mat::zeros(a1.size(), a1.type());
    for(int i = 1; i <=360; i++)
    {
        /// rotate with respect to the center using warpAffine
        double angle_transform = i;
        double scale_transform = 1.4;
        Mat rot_mat_transform( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform = getRotationMatrix2D( ICenter, angle_transform+90, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a, b, rot_mat_transform, a.size() );

        /// rotate with respect to the translated point (rCentre_a1) using warpAffine
        Mat rot_mat_transform1( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform1 = getRotationMatrix2D( rCentre_a1, angle_transform, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a1, b1, rot_mat_transform1, a1.size() );

        /// rotate with respect to the translated point using RotatedRect
        Mat b2 = Mat::zeros(a.size(), a.type());
        RotatedRect rRect_b2 = RotatedRect(rCentre_a1, Size2f(40,20), -90-i);
        Point2f vertices_b2[3];
        rRect_b2.points(vertices_b2);
        for (int i = 0; i < 3; i++)
            line(b2, vertices_b2[i], vertices_b2[(i+1)%3], Scalar(255));

        /// show the test images
        if (a.empty() || b.empty() || a1.empty() || b1.empty() || b2.empty())
            return -1;

        /** magnitude of DFT */
        FemtoTrack femtoTrack;
        Mat magI = femtoTrack.magnitudeDFT(b);
        normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                                // viewable image form (float between values 0 and 1).
        Mat magI1 = femtoTrack.magnitudeDFT(b1);
        normalize(magI1, magI1, 0, 1, CV_MINMAX);

        Mat magI2 = femtoTrack.magnitudeDFT(b2);
        normalize(magI2, magI2, 0, 1, CV_MINMAX);

        imshow("a"    , a    );
        imshow("a1"   , a1   );
        imshow("b"    , b    );    // Show the result
        imshow("magI" , magI );
        imshow("b1"   , b1   );    // Show the result
        imshow("magI1", magI1);
        imshow("b2"   , b2   );    // Show the result
        imshow("magI2", magI2);
         //waitKey();
        waitKey(10);
    }
    return 0;
}

int DemoDFT::demoLogPolar()
{
    // generate two test images
//    Mat a(200, 200, CV_8UC3, Scalar(0));
//    Mat b(200, 200, CV_8UC3, Scalar(0));
    Mat a(200, 200, CV_8UC1, Scalar(0));
    Mat b(200, 200, CV_8UC1, Scalar(0));

    RotatedRect rRect_a = RotatedRect(Point2f(100,100), Size2f(100,50), 0);
    RotatedRect rRect_b = RotatedRect(Point2f(100,100), Size2f(100,50), 30);

    Point2f vertices_a[4];
    rRect_a.points(vertices_a);
    for (int i = 0; i < 3; i++)
//        line(a, vertices_a[i], vertices_a[(i+1)%4], Scalar(0,0,255));
        line(a, vertices_a[i], vertices_a[(i+1)%3], Scalar(255));

    Point2f vertices_b[4];
    rRect_b.points(vertices_b);
    for (int i = 0; i < 3; i++)
//        line(b, vertices_b[i], vertices_b[(i+1)%4], Scalar(0,0,255));
        line(b, vertices_b[i], vertices_b[(i+1)%3], Scalar(255));

    // FFT test
    if (a.empty() || b.empty())
        return -1;

    imshow("a", a);
    imshow("b", b);
//    imwrite( "G:\\wenhao.fu\\Searches\\Code\\Registration\\FFTLogPolar\\a.png", a );
//    imwrite( "G:\\wenhao.fu\\Searches\\Code\\Registration\\FFTLogPolar\\b.png", b);


//    Mat a = cv::imread("G:\\wenhao.fu\\Searches\\Code\\Registration\\FFTLogPolar\\a.png", 0);
//    Mat b = cv::imread("G:\\wenhao.fu\\Searches\\Code\\Registration\\FFTLogPolar\\b.png", 0);
//    if (a.empty() || b.empty())
//        return -1;
//
//    imshow("a", a);
//    imshow("b", b);

    Mat pa = Mat::zeros(a.size(), CV_8UC1);
    Mat pb = Mat::zeros(b.size(), CV_8UC1);
    IplImage ipl_a = a, ipl_pa = pa;
    IplImage ipl_b = b, ipl_pb = pb;
    cvLogPolar(&ipl_a, &ipl_pa, cvPoint2D32f(a.cols >> 1, a.rows >> 1), _m);
    cvLogPolar(&ipl_b, &ipl_pb, cvPoint2D32f(b.cols >> 1, b.rows >> 1), _m);

    imshow("logpolar a", pa);
    imshow("logpolar b", pb);

    Mat pa_64f, pb_64f;
    pa.convertTo(pa_64f, CV_64F);
    pb.convertTo(pb_64f, CV_64F);

    Point2d pt = phaseCorrelate(pa_64f, pb_64f);

    std::cout << "Rotation = " << cv::format("%.2f", pt.y*180/(a.cols >> 1))
              << std::endl;

    std::cout << "Scale = " << exp(pt.x/_m) <<std::endl;

    std::cout << "done " << std::endl;
    ///waitKey(0);


    /** recover the image */

    /// Compute a rotation matrix with respect to the center of the image
    Point center_recover = Point( b.cols/2, b.rows/2 );
    double angle_recover = pt.y*180/(a.cols >> 1);
    double scale_recover = 1.f/exp(pt.x/_m);
    Mat rot_mat_recover( 2, 3, CV_32FC1 );

    /// Set the dst image the same type and size as src
    Mat b_recover = Mat::zeros(b.size(), b.type());

    /// Get the rotation matrix with the specifications above
    rot_mat_recover = getRotationMatrix2D( center_recover, angle_recover, scale_recover );

    /// de-Rotating and de-scale the image b
    /// Rotate the warped image
    warpAffine( b, b_recover, rot_mat_recover, b.size() );

    /// Show what you got
    namedWindow( "b_recover", WINDOW_AUTOSIZE );
    imshow( "b_recover", b_recover );

    /// Wait until user exits the program
    waitKey(0);

    return 0;
}

int DemoDFT::demoRotMagDFTLogPolar()
{
    /** generate a reference image */
    /// generate a figure in the image in the center
    Mat a(200, 200, CV_8UC1, Scalar(0));
    Point2f ICenter(a.cols/2,a.rows/2);
    RotatedRect rRect_a = RotatedRect(ICenter, Size2f(40,20), 0);
    Point2f vertices_a[3];
    rRect_a.points(vertices_a);
    for (int i = 0; i < 3; i++)
        line(a, vertices_a[i], vertices_a[(i+1)%3], Scalar(255));

    /// generate a figure in the image with a translation
    Mat a1(200, 200, CV_8UC1, Scalar(0));
    Point2f rCentre_a1 = ICenter + Point2f(60,-60);  // object center
    RotatedRect rRect_a1 = RotatedRect(rCentre_a1, Size2f(40,20), -90);
    Point2f vertices_a1[3];
    rRect_a1.points(vertices_a1);
    for (int i = 0; i < 3; i++)
        line(a1, vertices_a1[i], vertices_a1[(i+1)%3], Scalar(255));

    /// generate the rotated image and do magnitude of DFT
    for(int i = 1; i <=360; i++)
    {
        /// rotate with respect to the center using warpAffine
        Mat b = Mat::zeros(a.size(), a.type()); // rotate with
        double angle_transform = i;
        double scale_transform = 1.0;
        Mat rot_mat_transform( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform = getRotationMatrix2D( ICenter, angle_transform+90, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a, b, rot_mat_transform, a.size() );

        /// rotate with respect to the translated point (rCentre_a1) using warpAffine
        Mat b1 = Mat::zeros(a1.size(), a1.type());
        Mat rot_mat_transform1( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform1 = getRotationMatrix2D( rCentre_a1, angle_transform, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a1, b1, rot_mat_transform1, a1.size() );

        /// rotate with respect to the translated point using RotatedRect
        Mat b2 = Mat::zeros(a.size(), a.type());
        RotatedRect rRect_b2 = RotatedRect(rCentre_a1, Size2f(40,20), -90-i);
        Point2f vertices_b2[3];
        rRect_b2.points(vertices_b2);
        for (int i = 0; i < 3; i++)
            line(b2, vertices_b2[i], vertices_b2[(i+1)%3], Scalar(255));

        /// rotate with respect to the center with translation from the center using warpAffine
        Mat b3 = Mat::zeros(a1.size(), a1.type());
        Mat rot_mat_transform3( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform3 = getRotationMatrix2D( ICenter, angle_transform, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a1, b3, rot_mat_transform3, a1.size() );

        /// show the test images
        if (a.empty() || b.empty() || a1.empty() || b1.empty() || b2.empty())
            return -1;

        /** magnitude of DFT */
        FemtoTrack femtoTrack;
        Mat magI  = femtoTrack.magnitudeDFT(b );
        Mat magI1 = femtoTrack.magnitudeDFT(b1);
        Mat magI2 = femtoTrack.magnitudeDFT(b2);
        Mat magI3 = femtoTrack.magnitudeDFT(b3);

        /** Log-Polar of spectrum magnitude (magnitude of DFT) */
        Mat lpMagI  = femtoTrack.logPolar(magI );
        Mat lpMagI1 = femtoTrack.logPolar(magI1);
        Mat lpMagI2 = femtoTrack.logPolar(magI2);
        Mat lpMagI3 = femtoTrack.logPolar(magI3);

        /** show the results */
        /// Transform the matrix with float values into a
        /// viewable image form (float between values 0 and 1).
        Mat showMagI   = Mat::zeros(magI.size()   , magI.type()   );
        normalize(magI   , showMagI   , 0, 1, CV_MINMAX);
        Mat showMagI1  = Mat::zeros(magI1.size()  , magI1.type()  );
        normalize(magI1  , showMagI1  , 0, 1, CV_MINMAX);
        Mat showMagI2  = Mat::zeros(magI2.size()  , magI2.type()  );
        normalize(magI2  , showMagI2  , 0, 1, CV_MINMAX);
        Mat showMagI3  = Mat::zeros(magI3.size()  , magI3.type()  );
        normalize(magI3  , showMagI3  , 0, 1, CV_MINMAX);
        Mat showlpMagI = Mat::zeros(lpMagI.size() , lpMagI.type() );
        normalize(lpMagI , showlpMagI , 0, 1, CV_MINMAX);
        Mat showlpMagI1= Mat::zeros(lpMagI1.size(), lpMagI1.type());
        normalize(lpMagI1, showlpMagI1, 0, 1, CV_MINMAX);
        Mat showlpMagI2= Mat::zeros(lpMagI2.size(), lpMagI2.type());
        normalize(lpMagI2, showlpMagI2, 0, 1, CV_MINMAX);
        Mat showlpMagI3= Mat::zeros(lpMagI3.size(), lpMagI3.type());
        normalize(lpMagI3, showlpMagI3, 0, 1, CV_MINMAX);

        imshow("a"      , a          );
        imshow("a1"     , a1         );
        imshow("b"      , b          );    // Show the result
        imshow("magI"   , showMagI   );
        imshow("lpMagI" , showlpMagI );
        imshow("b1"     , b1         );    // Show the result
        imshow("magI1"  , showMagI1  );
        imshow("lpMagI1", showlpMagI1);
        imshow("b2"     , b2         );    // Show the result
        imshow("magI2"  , showMagI2  );
        imshow("lpMagI2", showlpMagI2);
        imshow("b3"     , b3         );    // Show the result
        imshow("magI3"  , showMagI3  );
        imshow("lpMagI3", showlpMagI3);
         //waitKey();
        waitKey(10);
    }
    return 0;
}

int DemoDFT::demoRotMagDFTLogPolarPC()
{
    /** generate a reference image */
    /// generate a figure in the image in the center
    Mat a(200, 200, CV_8UC1, Scalar(0));
    Point2f ICenter(a.cols/2,a.rows/2);
    RotatedRect rRect_a = RotatedRect(ICenter, Size2f(40,20), 0);
    Point2f vertices_a[3];
    rRect_a.points(vertices_a);
    for (int i = 0; i < 3; i++)
        line(a, vertices_a[i], vertices_a[(i+1)%3], Scalar(255));

    /// generate a figure in the image with a translation
    Mat a1(200, 200, CV_8UC1, Scalar(0));
    Point2f rCentre_a1 = ICenter + Point2f(60,-60);  // object center
    RotatedRect rRect_a1 = RotatedRect(rCentre_a1, Size2f(40,20), -90);
    Point2f vertices_a1[3];
    rRect_a1.points(vertices_a1);
    for (int i = 0; i < 3; i++)
        line(a1, vertices_a1[i], vertices_a1[(i+1)%3], Scalar(255));

    /// generate the rotated image and do magnitude of DFT
    for(int i = 1; i <=360; i++)
    {
        /// rotate with respect to the center using warpAffine
        Mat b = Mat::zeros(a.size(), a.type()); // rotate with
        double angle_transform = i;
        double scale_transform = 1.0;
        Mat rot_mat_transform( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform = getRotationMatrix2D( ICenter, angle_transform+90, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a, b, rot_mat_transform, a.size() );

        /// rotate with respect to the translated point (rCentre_a1) using warpAffine
        Mat b1 = Mat::zeros(a1.size(), a1.type());
        Mat rot_mat_transform1( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform1 = getRotationMatrix2D( rCentre_a1, angle_transform, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a1, b1, rot_mat_transform1, a1.size() );

        /// rotate with respect to the translated point using RotatedRect
        Mat b2 = Mat::zeros(a.size(), a.type());
        RotatedRect rRect_b2 = RotatedRect(rCentre_a1, Size2f(40,20), -90-i);
        Point2f vertices_b2[3];
        rRect_b2.points(vertices_b2);
        for (int i = 0; i < 3; i++)
            line(b2, vertices_b2[i], vertices_b2[(i+1)%3], Scalar(255));

        /// rotate with respect to the center with translation from the center using warpAffine
        Mat b3 = Mat::zeros(a1.size(), a1.type());
        Mat rot_mat_transform3( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform3 = getRotationMatrix2D( ICenter, angle_transform, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a1, b3, rot_mat_transform3, a1.size() );

        /// show the test images
        if (a.empty() || b.empty() || a1.empty() || b1.empty() || b2.empty())
            return -1;

        /** magnitude of DFT */
        FemtoTrack femtoTrack;
        Mat magI  = femtoTrack.magnitudeDFT(b );
        Mat magI1 = femtoTrack.magnitudeDFT(b1);
        Mat magI2 = femtoTrack.magnitudeDFT(b2);
        Mat magI3 = femtoTrack.magnitudeDFT(b3);

        /** Log-Polar of spectrum magnitude (magnitude of DFT) */
        Mat lpMagI  = femtoTrack.logPolar(magI );
        Mat lpMagI1 = femtoTrack.logPolar(magI1);
        Mat lpMagI2 = femtoTrack.logPolar(magI2);
        Mat lpMagI3 = femtoTrack.logPolar(magI3);

        /** show the results */
        /// Transform the matrix with float values into a
        /// viewable image form (float between values 0 and 1).
        Mat showMagI   = Mat::zeros(magI.size()   , magI.type()   );
        normalize(magI   , showMagI   , 0, 1, CV_MINMAX);
        Mat showMagI1  = Mat::zeros(magI1.size()  , magI1.type()  );
        normalize(magI1  , showMagI1  , 0, 1, CV_MINMAX);
        Mat showMagI2  = Mat::zeros(magI2.size()  , magI2.type()  );
        normalize(magI2  , showMagI2  , 0, 1, CV_MINMAX);
        Mat showMagI3  = Mat::zeros(magI3.size()  , magI3.type()  );
        normalize(magI3  , showMagI3  , 0, 1, CV_MINMAX);
        Mat showlpMagI = Mat::zeros(lpMagI.size() , lpMagI.type() );
        normalize(lpMagI , showlpMagI , 0, 1, CV_MINMAX);
        Mat showlpMagI1= Mat::zeros(lpMagI1.size(), lpMagI1.type());
        normalize(lpMagI1, showlpMagI1, 0, 1, CV_MINMAX);
        Mat showlpMagI2= Mat::zeros(lpMagI2.size(), lpMagI2.type());
        normalize(lpMagI2, showlpMagI2, 0, 1, CV_MINMAX);
        Mat showlpMagI3= Mat::zeros(lpMagI3.size(), lpMagI3.type());
        normalize(lpMagI3, showlpMagI3, 0, 1, CV_MINMAX);

        imshow("a"      , a          );
        imshow("a1"     , a1         );
        imshow("b"      , b          );    // Show the result
        imshow("magI"   , showMagI   );
        imshow("lpMagI" , showlpMagI );
        imshow("b1"     , b1         );    // Show the result
        imshow("magI1"  , showMagI1  );
        imshow("lpMagI1", showlpMagI1);
        imshow("b2"     , b2         );    // Show the result
        imshow("magI2"  , showMagI2  );
        imshow("lpMagI2", showlpMagI2);
        imshow("b3"     , b3         );    // Show the result
        imshow("magI3"  , showMagI3  );
        imshow("lpMagI3", showlpMagI3);
         //waitKey();
        waitKey(10);
    }
    return 0;
}

/// the first correlation : pure rotation with respect to the image center
/// the second correlation: translation and rotation with respect to the image center
/// the third correlation : translation and rotation with respect to the object center
int DemoDFT::demoMagDFTLogPolarPCSimu()
{
    /*** generate a reference image */
    /// generate a figure in the image in the center
    Mat a(200, 200, CV_8UC1, Scalar(0));
    Point2f ICenter(a.cols/2,a.rows/2);
    RotatedRect rRect_a = RotatedRect(ICenter, Size2f(30,20), 0);
    Point2f vertices_a[3];
    rRect_a.points(vertices_a);
    for (int i = 0; i < 3; i++)
        line(a, vertices_a[i], vertices_a[(i+1)%3], Scalar(255));

    Point2f translation(60,-60);
    double angle_transform = 60;
    double scale_transform = 1.0;
    /// generate a figure in the image with a translation
    Mat a1(200, 200, CV_8UC1, Scalar(0));
    Point2f rCentre_a1 = ICenter + translation;  // object center
    RotatedRect rRect_a1 = RotatedRect(rCentre_a1, Size2f(30,20), 0);
    Point2f vertices_a1[3];
    rRect_a1.points(vertices_a1);
    for (int i = 0; i < 3; i++)
        line(a1, vertices_a1[i], vertices_a1[(i+1)%3], Scalar(255));

    /// generate a figure in the image with a pure rotation with respect to the image center
    Mat a2(200, 200, CV_8UC1, Scalar(0));
    Mat rot_mat_transform0( 2, 3, CV_32FC1 );
    /// Get the rotation matrix with the specifications above
    rot_mat_transform0 = getRotationMatrix2D( ICenter, angle_transform, scale_transform );
    /// transform the image a to get the transformed image b
    warpAffine( a, a2, rot_mat_transform0, a.size() );

    /** generate the rotated images and do magnitude of DFT */
    /// pure rotate with respect to the image center using warpAffine
    Mat b = Mat::zeros(a.size(), a.type()); // rotate with
    Mat rot_mat_transform( 2, 3, CV_32FC1 );
    /// Get the rotation matrix with the specifications above
    //rot_mat_transform = getRotationMatrix2D( ICenter, angle_transform+90, scale_transform );
    rot_mat_transform = getRotationMatrix2D( ICenter, angle_transform, scale_transform );
    /// transform the image a to get the transformed image b
    warpAffine( a, b, rot_mat_transform, a.size() );

    /// rotate with respect to the translated point (rCentre_a1) using warpAffine
    Mat b1 = Mat::zeros(a1.size(), a1.type());
    Mat rot_mat_transform1( 2, 3, CV_32FC1 );
    /// Get the rotation matrix with the specifications above
    rot_mat_transform1 = getRotationMatrix2D( rCentre_a1, angle_transform, scale_transform );
    /// transform the image a to get the transformed image b
    warpAffine( a1, b1, rot_mat_transform1, a1.size() );

    /// rotate with respect to the translated point using RotatedRect
    Mat b2 = Mat::zeros(a.size(), a.type());
    RotatedRect rRect_b2 = RotatedRect(rCentre_a1, Size2f(30,20), -angle_transform);
    Point2f vertices_b2[3];
    rRect_b2.points(vertices_b2);
    for (int i = 0; i < 3; i++)
        line(b2, vertices_b2[i], vertices_b2[(i+1)%3], Scalar(255));

    /// translation and rotation with respect to the image center using warpAffine
    Mat b3 = Mat::zeros(a1.size(), a1.type());
    Mat rot_mat_transform3( 2, 3, CV_32FC1 );
    /// Get the rotation matrix with the specifications above
    rot_mat_transform3 = getRotationMatrix2D( ICenter, angle_transform, scale_transform );
    /// transform the image a to get the transformed image b
    warpAffine( a1, b3, rot_mat_transform3, a1.size() );

    /// show the test images
    if (a.empty() || b.empty() || a1.empty() || b1.empty() || b2.empty())
        return -1;

    /*** Image Registration */
    /** magnitude of DFT */
    FemtoTrack femtoTrack(50);
    Mat magI0 = femtoTrack.magnitudeDFT(a );
    Mat magI  = femtoTrack.magnitudeDFT(b );
    Mat magI1 = femtoTrack.magnitudeDFT(b1);
    Mat magI2 = femtoTrack.magnitudeDFT(b2);
    Mat magI3 = femtoTrack.magnitudeDFT(b3);

    /** High-Pass Filter Module */

    /** Log-Polar of spectrum magnitude (magnitude of DFT) */
    Mat lpMagI0 = femtoTrack.logPolar(magI0);
    Mat lpMagI  = femtoTrack.logPolar(magI );
    Mat lpMagI1 = femtoTrack.logPolar(magI1);
    Mat lpMagI2 = femtoTrack.logPolar(magI2);
    Mat lpMagI3 = femtoTrack.logPolar(magI3);

    /** Phase correlation Module */
    Mat lpmagI1_64f, lpmagI2_64f;
    /// pure rotate with respect to the image center using warpAffine
    lpMagI0.convertTo(lpmagI1_64f, CV_64F);
    lpMagI.convertTo(lpmagI2_64f, CV_64F);
    Point2d pt0 = phaseCorrelate(lpmagI1_64f, lpmagI2_64f);
    /// rotate with respect to the translated point (rCentre_a1) using warpAffine
    lpMagI1.convertTo(lpmagI2_64f, CV_64F);
    Point2d pt1 = phaseCorrelate(lpmagI1_64f, lpmagI2_64f);
    /// rotate with respect to the translated point using RotatedRect
    lpMagI2.convertTo(lpmagI2_64f, CV_64F);
    Point2d pt2 = phaseCorrelate(lpmagI1_64f, lpmagI2_64f);
    /// translation and rotation with respect to the image center using warpAffine
    lpMagI3.convertTo(lpmagI2_64f, CV_64F);
    Point2d pt3 = phaseCorrelate(lpmagI1_64f, lpmagI2_64f);

    /** Transformation Module */
    /// Re-scale and de-rotated image
    Mat rb3 = Mat::zeros(b3.size(), b3.type());
    Mat rb3_180 = Mat::zeros(b3.size(), b3.type());
    Mat rot_mat_transformRb3( 2, 3, CV_32FC1 );
    rot_mat_transformRb3 = getRotationMatrix2D( ICenter, -pt3.y*360/(a.cols), 1/exp(pt3.x/femtoTrack.read_m()));
    Mat rot_mat_transformRb3_180 = getRotationMatrix2D( ICenter, -pt3.y*360/(a.cols)+180, 1/exp(pt3.x/femtoTrack.read_m()));
    /// transform the image a to get the recovered image a from b3
    warpAffine( b3, rb3, rot_mat_transformRb3, b3.size(), WARP_INVERSE_MAP, BORDER_CONSTANT, 0 );
    warpAffine( b3, rb3_180, rot_mat_transformRb3_180, b3.size(), WARP_INVERSE_MAP, BORDER_CONSTANT, 0 );

    /** Phase correlation */
    Mat a_64f, rb3_64f, rb3_180_64f;
    a.convertTo(    a_64f, CV_64F);
    rb3.convertTo(rb3_64f, CV_64F);
    Point2d ptRb3 = phaseCorrelate(a_64f, rb3_64f);
    rb3_180.convertTo(rb3_180_64f, CV_64F);
    Point2d ptRb3_180 = phaseCorrelate(a_64f, rb3_180_64f);

    /** Translation Module */
    Mat rrb3 = Mat::zeros(b3.size(), b3.type());
    Mat rrb3_180 = Mat::zeros(b3.size(), b3.type());
    rrb3 = femtoTrack.translateImg(rb3, ptRb3.x, ptRb3.y);
    rrb3_180 = femtoTrack.translateImg(rb3_180, ptRb3_180.x, ptRb3_180.y);

    /*** show the results */
    /// Transform the matrix with float values into a
    /// viewable image form (float between values 0 and 1).
    Mat showMagI0  = Mat::zeros(magI0.size()  , magI0.type()   );
    normalize(magI0  , showMagI0  , 0, 1, CV_MINMAX);
    Mat showMagI   = Mat::zeros(magI.size()   , magI.type()   );
    normalize(magI   , showMagI   , 0, 1, CV_MINMAX);
    Mat showMagI1  = Mat::zeros(magI1.size()  , magI1.type()  );
    normalize(magI1  , showMagI1  , 0, 1, CV_MINMAX);
    Mat showMagI2  = Mat::zeros(magI2.size()  , magI2.type()  );
    normalize(magI2  , showMagI2  , 0, 1, CV_MINMAX);
    Mat showMagI3  = Mat::zeros(magI3.size()  , magI3.type()  );
    normalize(magI3  , showMagI3  , 0, 1, CV_MINMAX);
    Mat showlpMagI0= Mat::zeros(lpMagI0.size(), lpMagI0.type());
    normalize(lpMagI0, showlpMagI0, 0, 1, CV_MINMAX);
    Mat showlpMagI = Mat::zeros(lpMagI.size() , lpMagI.type() );
    normalize(lpMagI , showlpMagI , 0, 1, CV_MINMAX);
    Mat showlpMagI1= Mat::zeros(lpMagI1.size(), lpMagI1.type());
    normalize(lpMagI1, showlpMagI1, 0, 1, CV_MINMAX);
    Mat showlpMagI2= Mat::zeros(lpMagI2.size(), lpMagI2.type());
    normalize(lpMagI2, showlpMagI2, 0, 1, CV_MINMAX);
    Mat showlpMagI3= Mat::zeros(lpMagI3.size(), lpMagI3.type());
    normalize(lpMagI3, showlpMagI3, 0, 1, CV_MINMAX);

    imshow("a"       , a          );
    imshow("a1"      , a1         );
    imshow("a2"      , a2         );
    imshow("b"       , b          );    // Show the result
    imshow("magI0"   , showMagI0  );
    imshow("lpMagI0" , showlpMagI0);
    imshow("magI"    , showMagI   );
    imshow("lpMagI"  , showlpMagI );
    imshow("b1"      , b1         );    // Show the result
    imshow("magI1"   , showMagI1  );
    imshow("lpMagI1" , showlpMagI1);
    imshow("b2"      , b2         );    // Show the result
    imshow("magI2"   , showMagI2  );
    imshow("lpMagI2" , showlpMagI2);
    imshow("b3"      , b3         );    // Show the result
    imshow("magI3"   , showMagI3  );
    imshow("lpMagI3" , showlpMagI3);
    imshow("rb3"     , rb3        );
    imshow("rb3_180" , rb3_180    );
    imshow("rrb3"    , rrb3       );
    imshow("rrb3_180", rrb3_180   );

    /// the registrated transformation: rotation, scale and translation
    std::cout << "Simulation initialization: " << std::endl;
    std::cout << "Rotation = " << angle_transform << std::endl;
    std::cout << "Scale = " << scale_transform << std::endl;
    std::cout << "Translation x = " << translation.x << std::endl;
    std::cout << "Translation y = " << translation.y << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "pure rotate with respect to the image center using warpAffine " << std::endl;
    std::cout << "Rotation = " << cv::format("%.2f", pt0.y*180/(a.cols >> 1))
              << std::endl;
    std::cout << "Scale = " << exp(pt0.x/femtoTrack.read_m()) << std::endl;
    std::cout << "m = " << femtoTrack.read_m() << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "rotate with respect to the translated point (rCentre_a1) using warpAffine " << std::endl;
    std::cout << "Rotation = " << cv::format("%.2f", pt1.y*180/(a.cols >> 1))
              << std::endl;
    std::cout << "Scale = " << exp(pt1.x/femtoTrack.read_m()) << std::endl;
    std::cout << "m = " << femtoTrack.read_m() << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "rotate with respect to the translated point using RotatedRect " << std::endl;
    std::cout << "Rotation = " << cv::format("%.2f", pt2.y*180/(a.cols >> 1))
              << std::endl;
    std::cout << "Scale = " << exp(pt2.x/femtoTrack.read_m()) << std::endl;
    std::cout << "m = " << femtoTrack.read_m() << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "translation and rotation with respect to the image center using warpAffine " << std::endl;
    std::cout << "Rotation = " << cv::format("%.2f", pt3.y*180/(a.cols >> 1))
              << std::endl;
    std::cout << "Scale = " << exp(pt3.x/femtoTrack.read_m()) << std::endl;
    std::cout << "Translation x = " << ptRb3.x << std::endl;
    std::cout << "Translation y = " << ptRb3.y << std::endl;
    std::cout << "Translation x (+180)= " << ptRb3_180.x << std::endl;
    std::cout << "Translation y (+180)= " << ptRb3_180.y << std::endl;
    std::cout << "m = " << femtoTrack.read_m() << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    waitKey();

    return 0;
}

int DemoDFT::demoMagDFTLogPolarPCImage()
{
//    /*** read color images */
//    string filename = "G:\\wenhao.fu\\Research\\Code\\Robotex\\SEM0.png";
//    Mat image = imread(filename, 1);
//    Mat a, b;
//    a.create(image.size(), image.type());
//    cvtColor(image, a, COLOR_BGR2GRAY);

    /*** read gray scale images */
    Mat a = imread("lena.jpg", 0);
    resize(a, a, Size(255,255), 0, 0, CV_INTER_AREA );
    /// generate a rotated image
    Point2f ICenter(a.cols/2,a.rows/2);
    double angle_transform = 80;
    double scale_transform = 1.0;
    /// translation and rotation with respect to the image center using warpAffine
    Mat b3 = Mat::zeros(a.size(), a.type());
    Mat rot_mat_transform3( 2, 3, CV_32FC1 );
    /// Get the rotation matrix with the specifications above
    rot_mat_transform3 = getRotationMatrix2D( ICenter, angle_transform, scale_transform );
    /// transform the image a to get the transformed image b
    warpAffine( a, b3, rot_mat_transform3, a.size(), INTER_CUBIC );


//    Mat b3 = imread("G:\\wenhao.fu\\Research\\Code\\Robotex\\image.0501.pgm", 0);
//    Mat a = imread("C:\\ViSP-2.10.0\\ViSP-images\\mire-2\\mire.pgm", 1);
//    Mat b3 = imread("C:\\ViSP-2.10.0\\ViSP-images\\mire-2\\image.0001.pgm", 1);
    if (a.empty() || b3.empty())
        return -1;

    //resize(a, a, b3.size(), 0, 0, CV_INTER_AREA );

    waitKey();

    /*** Canny Edge Detector */
//    Mat a  = femtoTrack.CannyThreshold(image1);
//    Mat b3 = femtoTrack.CannyThreshold(image2);

    /*** Image Registration */
    //Point2f ICenter(a.cols/2,a.rows/2);
    FemtoTrack femtoTrack(95);
    //double angle_transform, scale_transform;
    double x_transform, y_transform;
    /** magnitude of DFT */
    Mat magI0 = femtoTrack.magnitudeDFT(a );
    Mat magI3 = femtoTrack.magnitudeDFT(b3);

    /** High-Pass Filter Module */

    /** Log-Polar of spectrum magnitude (magnitude of DFT) */
    Mat lpMagI0 = femtoTrack.logPolar(magI0);
    Mat lpMagI3 = femtoTrack.logPolar(magI3);

//    /** Phase correlation Module */
//    Mat lpmagI1_64f, lpmagI2_64f;
//    lpMagI0.convertTo(lpmagI1_64f, CV_64F);
//    lpMagI3.convertTo(lpmagI2_64f, CV_64F);
//    Point2d pt3 = phaseCorrelate(lpmagI1_64f, lpmagI2_64f);

    /** Phase correlation Module */
    Mat lpmagI1_64f, lpmagI2_64f;
    lpMagI0.convertTo(lpmagI1_64f, CV_64F);
    lpMagI3.convertTo(lpmagI2_64f, CV_64F);
    Point2d pt3 = phaseCorrelate(lpmagI1_64f, lpmagI2_64f);

    /** Transformation Module */
    /// Re-scale and de-rotated image
    Mat rb3 = Mat::zeros(b3.size(), b3.type());
    Mat rb3_180 = Mat::zeros(b3.size(), b3.type());
    Mat rot_mat_transformRb3( 2, 3, CV_32FC1 );
    rot_mat_transformRb3 = getRotationMatrix2D( ICenter, -pt3.y*360/(a.cols), 1/exp(pt3.x/femtoTrack.read_m()));
    Mat rot_mat_transformRb3_180 = getRotationMatrix2D( ICenter, -pt3.y*360/(a.cols)+180, 1/exp(pt3.x/femtoTrack.read_m()));
    /// transform the image a to get the recovered image a from b3
    warpAffine( b3, rb3, rot_mat_transformRb3, b3.size(), WARP_INVERSE_MAP, BORDER_CONSTANT, 0 );
    warpAffine( b3, rb3_180, rot_mat_transformRb3_180, b3.size(), WARP_INVERSE_MAP, BORDER_CONSTANT, 0 );

    /** Phase correlation */
    Mat a_64f, rb3_64f, rb3_180_64f;
    a.convertTo(    a_64f, CV_64F);
    rb3.convertTo(rb3_64f, CV_64F);
    Point2d ptRb3 = phaseCorrelate(a_64f, rb3_64f);
    rb3_180.convertTo(rb3_180_64f, CV_64F);
    Point2d ptRb3_180 = phaseCorrelate(a_64f, rb3_180_64f);

    /** Translation Module */
    Mat rrb3 = Mat::zeros(b3.size(), b3.type());
    Mat rrb3_180 = Mat::zeros(b3.size(), b3.type());
    rrb3 = femtoTrack.translateImg(rb3, ptRb3.x, ptRb3.y);
    rrb3_180 = femtoTrack.translateImg(rb3_180, ptRb3_180.x, ptRb3_180.y);

    /*** show the results */
    /// Transform the matrix with float values into a
    /// viewable image form (float between values 0 and 1).
    Mat showMagI0  = Mat::zeros(magI0.size()  , magI0.type()   );
    normalize(magI0  , showMagI0  , 0, 1, CV_MINMAX);
    Mat showMagI3  = Mat::zeros(magI3.size()  , magI3.type()  );
    normalize(magI3  , showMagI3  , 0, 1, CV_MINMAX);
    Mat showlpMagI0= Mat::zeros(lpMagI0.size(), lpMagI0.type());
    normalize(lpMagI0, showlpMagI0, 0, 1, CV_MINMAX);
    Mat showlpMagI3= Mat::zeros(lpMagI3.size(), lpMagI3.type());
    normalize(lpMagI3, showlpMagI3, 0, 1, CV_MINMAX);

    namedWindow( "a"        , CV_WINDOW_AUTOSIZE );
    namedWindow( "magI0"    , CV_WINDOW_AUTOSIZE );
    namedWindow( "lpMagI0"  , CV_WINDOW_AUTOSIZE );
    namedWindow( "b3"       , CV_WINDOW_AUTOSIZE );
    namedWindow( "magI3"    , CV_WINDOW_AUTOSIZE );
    namedWindow( "lpMagI3"  , CV_WINDOW_AUTOSIZE );
    namedWindow( "rb3"      , CV_WINDOW_AUTOSIZE );
    namedWindow( "rb3_180"  , CV_WINDOW_AUTOSIZE );
    namedWindow( "rrb3"     , CV_WINDOW_AUTOSIZE );
    namedWindow( "rrb3_180" , CV_WINDOW_AUTOSIZE );
    int delta = 10;
    moveWindow(     "a  " , 0  , 0);
    moveWindow( "magI0  " , a.cols, 0);
    moveWindow( "lpMagI0" , a.cols+showMagI0.cols, 0);
    moveWindow(     "b3 " , 0  , a.rows+delta);
    moveWindow( "magI3  " , b3.cols, a.rows);
    moveWindow( "lpMagI3" , b3.cols+showMagI3.cols, a.rows);
    moveWindow( "rb3"      , 0  , 0 );
    moveWindow( "rb3_180"  , a.cols, 0 );
    moveWindow( "rrb3"     , 0  , a.rows );
    moveWindow( "rrb3_180" , b3.cols, a.rows );
    imshow("a"       , a          );
    imshow("magI0"   , showMagI0  );
    imshow("lpMagI0" , showlpMagI0);
    imshow("b3"      , b3         );    // Show the result
    imshow("magI3"   , showMagI3  );
    imshow("lpMagI3" , showlpMagI3);
    imshow("rb3"     , rb3        );
    imshow("rb3_180" , rb3_180    );
    imshow("rrb3"    , rrb3       );
    imshow("rrb3_180", rrb3_180   );

    /// Linear Adding (blending) two images a and rrb3
    Mat arrb3, arrb3_180;
    double alpha = 0.5; double beta;
    beta = ( 1.0 - alpha );
    addWeighted( a, alpha, rrb3, beta, 0.0, arrb3);
    addWeighted( a, alpha, rrb3_180, beta, 0.0, arrb3_180);

    namedWindow("arrb3", CV_WINDOW_AUTOSIZE);
    namedWindow("arrb3_180", CV_WINDOW_AUTOSIZE);
    imshow( "arrb3", arrb3 );
    imshow( "arrb3_180", arrb3_180 );


    /// the registrated transformation: rotation, scale and translation
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "translation and rotation with respect to the image center using warpAffine " << std::endl;
    std::cout << "Rotation = " << cv::format("%.2f", pt3.y*180/(a.cols >> 1))
              << std::endl;
    std::cout << "Scale = " << exp(pt3.x/femtoTrack.read_m()) << std::endl;
    std::cout << "Translation x = " << ptRb3.x << std::endl;
    std::cout << "Translation y = " << ptRb3.y << std::endl;
    std::cout << "Translation x (+180)= " << ptRb3_180.x << std::endl;
    std::cout << "Translation y (+180)= " << ptRb3_180.y << std::endl;
    std::cout << "m = " << femtoTrack.read_m() << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    waitKey();

    return 0;
}

/// the correlation : translation and rotation with respect to the object center
int DemoDFT::demoMagDFTLogPolarPCSimuVideo()
{
    /** generate a reference image */
    /// generate a figure in the image in the center
    Mat a(200, 200, CV_8UC1, Scalar(0));
    Point2f ICenter(a.cols/2,a.rows/2);
    RotatedRect rRect_a = RotatedRect(ICenter, Size2f(40,20), -90);
    //RotatedRect rRect_a = RotatedRect(ICenter, Size2f(40,20), 0);
    Point2f vertices_a[3];
    rRect_a.points(vertices_a);
    for (int i = 0; i < 3; i++)
        line(a, vertices_a[i], vertices_a[(i+1)%3], Scalar(255), 1);
        //line(a, vertices_a[i], vertices_a[(i+1)%3], Scalar(255));

    /// generate a figure in the image with a translation
    Point2f translation(60, -60);
    Mat a1(200, 200, CV_8UC1, Scalar(0));
    Point2f rCentre_a1 = ICenter + translation;  // object center
    RotatedRect rRect_a1 = RotatedRect(rCentre_a1, Size2f(40,20), -90);
    // RotatedRect rRect_a1 = RotatedRect(rCentre_a1, Size2f(40,20), 0);
    Point2f vertices_a1[3];
    rRect_a1.points(vertices_a1);
    for (int i = 0; i < 3; i++)
        line(a1, vertices_a1[i], vertices_a1[(i+1)%3], Scalar(255), 1);
        //line(a1, vertices_a1[i], vertices_a1[(i+1)%3], Scalar(255));

    /// generate the rotated image and do magnitude of DFT
    namedWindow( "a"        , CV_WINDOW_AUTOSIZE );
    namedWindow( "magI0"    , CV_WINDOW_AUTOSIZE );
    namedWindow( "lpMagI0"  , CV_WINDOW_AUTOSIZE );
    namedWindow( "b3"       , CV_WINDOW_AUTOSIZE );
    namedWindow( "magI3"    , CV_WINDOW_AUTOSIZE );
    namedWindow( "lpMagI3"  , CV_WINDOW_AUTOSIZE );
    namedWindow( "rb3"      , CV_WINDOW_AUTOSIZE );
    namedWindow( "rb3_180"  , CV_WINDOW_AUTOSIZE );
    namedWindow( "rrb3"     , CV_WINDOW_AUTOSIZE );
    namedWindow( "rrb3_180" , CV_WINDOW_AUTOSIZE );
    namedWindow( "rrrb3" , CV_WINDOW_AUTOSIZE );
    int delta = 10;
    moveWindow(     "a"   , delta  , delta);
    moveWindow( "magI0"   , a.cols+2*delta, delta);
    moveWindow( "lpMagI0" , 2*a.cols+4*delta, delta);
    moveWindow(     "b3"  , delta  , a.rows+4*delta);
    moveWindow( "magI3"   , a.cols+2*delta, a.rows+4*delta);
    moveWindow( "lpMagI3" , 2*a.cols+4*delta, a.rows+4*delta);
    moveWindow( "rb3"      , 3*a.cols+6*delta  , delta );
    moveWindow( "rrb3"     , 4*a.cols+8*delta  , delta );
    moveWindow( "rb3_180"  , 3*a.cols+6*delta, a.rows+4*delta );
    moveWindow( "rrb3_180" , 4*a.cols+8*delta, a.rows+4*delta );
    moveWindow( "rrrb3" , delta, 2*a.rows+6*delta );

    double angle_transform = 0;
    double scale_transform = 1.0;
    for(int i = 0; i <=360; i++)
    {
        angle_transform = i;
        /// translation and rotation with respect to the image center using warpAffine
        Mat b3 = Mat::zeros(a1.size(), a1.type());
        Mat rot_mat_transform3( 2, 3, CV_32FC1 );
        /// Get the rotation matrix with the specifications above
        rot_mat_transform3 = getRotationMatrix2D( ICenter, angle_transform, scale_transform );
        /// transform the image a to get the transformed image b
        warpAffine( a1, b3, rot_mat_transform3, a1.size() );

        /// show the test images
        // if (a.empty() || b.empty() || a1.empty() || b1.empty() || b2.empty())
        if (a.empty() || b3.empty())
            return -1;

        /*** Image Registration */
        /** magnitude of DFT */
        FemtoTrack femtoTrack(45);
        Mat magI0 = femtoTrack.magnitudeDFT(a );
        Mat magI3 = femtoTrack.magnitudeDFT(b3);

        /** High-Pass Filter Module */

        /** Log-Polar of spectrum magnitude (magnitude of DFT) */
        Mat lpMagI0 = femtoTrack.logPolar(magI0);
        Mat lpMagI3 = femtoTrack.logPolar(magI3);

        /** Phase correlation Module */
        Mat lpmagI1_64f, lpmagI2_64f;
        lpMagI0.convertTo(lpmagI1_64f, CV_64F);
        lpMagI3.convertTo(lpmagI2_64f, CV_64F);
        /// create a hanning Window
//        Mat hann;
//        createHanningWindow(hann, lpmagI1_64f.size(), CV_64F);
//        Point2d pt3 = phaseCorrelate(lpmagI1_64f, lpmagI2_64f, hann);
        Point2d pt3 = phaseCorrelate(lpmagI1_64f, lpmagI2_64f);

        /** Transformation Module */
        /// Compute angle of rotation and scale
        double theta, scale;
        theta = pt3.y*360/(a.cols);
        scale = exp(pt3.x/femtoTrack.read_m());
        /// Re-scale and Rotate image back by theta and theta + 180
        Mat rb3 = Mat::zeros(b3.size(), b3.type());
        Mat rb3_180 = Mat::zeros(b3.size(), b3.type());
        Mat rot_mat_transformRb3( 2, 3, CV_32FC1 );
        rot_mat_transformRb3 = getRotationMatrix2D( ICenter, theta, 1/scale );
        Mat rot_mat_transformRb3_180 = getRotationMatrix2D( ICenter, (theta+180), 1/scale);
        /// transform the image a to get the recovered image a from b3
        warpAffine( b3, rb3, rot_mat_transformRb3, b3.size(), INTER_CUBIC, BORDER_CONSTANT, 0 );
        warpAffine( b3, rb3_180, rot_mat_transformRb3_180, b3.size(), INTER_CUBIC, BORDER_CONSTANT, 0 );

        /** Phase correlation */
        Point2d ptT, ptT_180;  double pT, pT_180;
        Point2d ptTrans;       double angleTrans;   /// the affine results
        // Mat rrb3 = Mat::zeros(b3.size(), b3.type());

        Mat a_64f, rb3_64f, rb3_180_64f;
        a.convertTo(    a_64f, CV_64F);
        rb3.convertTo(rb3_64f, CV_64F);
        Point2d ptRb3 = phaseCorrelate(a_64f, rb3_64f);
        rb3_180.convertTo(rb3_180_64f, CV_64F);
        Point2d ptRb3_180 = phaseCorrelate(a_64f, rb3_180_64f);
        Mat emptyWindow;    // generate an empty matrix for no hanning window
        ptT = phaseCorrelateRes(a_64f, rb3_64f, emptyWindow, &pT);
        ptT_180 = phaseCorrelateRes(a_64f, rb3_180_64f, emptyWindow, &pT_180);
//        ptT = femtoTrack.phaseCorrelateRes(a_64f, rb3_64f, emptyWindow, &pT);
//        ptT_180 = femtoTrack.phaseCorrelateRes(a_64f, rb3_180_64f, emptyWindow, &pT_180);

        Mat rrrb3 = Mat::zeros(b3.size(), b3.type());

        if (pT > pT_180)
        {
            ptTrans = ptT;
            angleTrans = theta;
            rrrb3 = rb3;
        }
        else
        {
            ptTrans = ptT_180;
            angleTrans = angleNormalization(theta+ 180);
            rrrb3 = rb3_180;
        }

        /** Translation Module */
        Mat rrb3 = Mat::zeros(b3.size(), b3.type());
        Mat rrb3_180 = Mat::zeros(b3.size(), b3.type());
        rrb3 = femtoTrack.translateImg(rb3, -ptRb3);
        rrb3_180 = femtoTrack.translateImg(rb3_180, -ptRb3_180);
        rrrb3 = femtoTrack.translateImg(rrrb3, -ptTrans);

        /*** show the results */
        /// Transform the matrix with float values into a
        /// viewable image form (float between values 0 and 1).
        Mat showMagI0  = Mat::zeros(magI0.size()  , magI0.type()   );
        normalize(magI0  , showMagI0  , 0, 1, CV_MINMAX);
        Mat showMagI3  = Mat::zeros(magI3.size()  , magI3.type()  );
        normalize(magI3  , showMagI3  , 0, 1, CV_MINMAX);
        Mat showlpMagI0= Mat::zeros(lpMagI0.size(), lpMagI0.type());
        normalize(lpMagI0, showlpMagI0, 0, 1, CV_MINMAX);
        Mat showlpMagI3= Mat::zeros(lpMagI3.size(), lpMagI3.type());
        normalize(lpMagI3, showlpMagI3, 0, 1, CV_MINMAX);

        imshow("a"       , a          );
        imshow("magI0"   , showMagI0  );
        imshow("lpMagI0" , showlpMagI0);
        imshow("b3"      , b3         );    // Show the result
        imshow("magI3"   , showMagI3  );
        imshow("lpMagI3" , showlpMagI3);
        imshow("rb3"     , rb3        );
        imshow("rb3_180" , rb3_180    );
        imshow("rrb3"    , rrb3       );
        imshow("rrb3_180", rrb3_180   );
        imshow("rrrb3", rrrb3 );

        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "degree    :" << i  << std::endl;
        std::cout << "Scale     :" << format("%.2f", scale) << std::endl;
        std::cout << "theta     :" << format("%.2f", theta)  << std::endl;
        std::cout << "theta+180 :" << format("%.2f", theta+180)  << std::endl;
        std::cout << "angleTrans:" << format("%.2f", angleTrans) << std::endl;
        std::cout << "trans_x   :" << format("%.2f", ptT.x)  << std::endl;
        std::cout << "trans_y   :" << format("%.2f", ptT.y) << std::endl;

//        /// the registrated transformation: rotation, scale and translation
//        std::cout << "Simulation initialization: " << std::endl;
//        std::cout << "Rotation = " << angle_transform << std::endl;
//        std::cout << "Scale = " << scale_transform << std::endl;
//        std::cout << "Translation x = " << translation.x << std::endl;
//        std::cout << "Translation y = " << translation.y << std::endl;
//        std::cout << "--------------------------------------------" << std::endl;
//        std::cout << "translation and rotation with respect to the image center using warpAffine " << std::endl;
//        std::cout << "Rotation = " << cv::format("%.2f", pt3.y*180/(a.cols >> 1))
//                  << std::endl;
//        std::cout << "Scale = " << exp(pt3.x/femtoTrack.read_m()) << std::endl;
//        std::cout << "Translation x = " << ptRb3.x << std::endl;
//        std::cout << "Translation y = " << ptRb3.y << std::endl;
//        std::cout << "Translation x (+180)= " << ptRb3_180.x << std::endl;
//        std::cout << "Translation y (+180)= " << ptRb3_180.y << std::endl;
//        std::cout << "m = " << femtoTrack.read_m() << std::endl;
//        std::cout << "--------------------------------------------" << std::endl;
        waitKey(50);
    }

    waitKey();
    return 0;
}

//int DemoDFT::demoMagDFTLogPolarPC()
//{
//    /** generate the reference image */
//    Mat I1(200, 200, CV_8UC1, Scalar(0));
//    Mat I2 = Mat::zeros(I1.size(), I1.type());
//
//    /// generate a figure in the image
//    RotatedRect rRect_I1 = RotatedRect(Point2f(I1.cols/2,I1.rows/2), Size2f(100,50), 0);
//    Point2f vertices_I1[3];
//    rRect_I1.points(vertices_I1);
//    for (int i = 0; i < 3; i++)
//        line(I1, vertices_I1[i], vertices_I1[(i+1)%3], Scalar(255));
//
//    /** generate the transformed image */
//    /// first step: make a translation by changing the position of the center point
//    Point2f rCentre(I1.cols/2+43, I1.rows/2-38);
//    RotatedRect rRect_b = RotatedRect(rCentre, Size2f(100,50), 30);
//    Point2f vertices_b[3];
//    rRect_b.points(vertices_b);
//    for (int i = 0; i < 3; i++)
//        line(I2, vertices_b[i], vertices_b[(i+1)%3], Scalar(255));
//
//    /// second step: make a rotation and scale with respect to the image center
//    /// by computing a rotation matrix
//    Point center_transform = Point( I2.cols/2, I2.rows/2 );
//    double angle_transform = 0;
//    double scale_transform = 1.0;
//    Mat rot_mat_transform( 2, 3, CV_32FC1 );
//    /// Get the rotation matrix with the specifications above
//    rot_mat_transform = getRotationMatrix2D( center_transform, angle_transform, scale_transform );
//    /// transform the image a to get the transformed image b
//    warpAffine( I2, I2, rot_mat_transform, I2.size() );
//
//    /// show the test images
//    if (I1.empty() || I2.empty())
//            return -1;
//
//    /** magnitude of DFT */
//    FemtoTrack femtoTrack;
//    Mat I1_64f, I2_64f;
//    I1.convertTo(I1_64f, CV_64FC1);
//    I2.convertTo(I2_64f, CV_64FC1);
//
//    Mat magI1 = femtoTrack.magnitudeDFT(I1_64f);
//    Mat magI2 = femtoTrack.magnitudeDFT(I2_64f);
//    magI1.convertTo(magI1, CV_64FC1);
//    magI2.convertTo(magI2, CV_64FC1);
//
//    /** Log-Polar of spectrum magnitude (magnitude of DFT) */
//    double M = 40;
//    Mat lpmagI1 = Mat::zeros(magI1.size(), CV_64FC1);
//    Mat lpmagI2 = Mat::zeros(magI2.size(), CV_64FC1);
//    IplImage ipl_magI1 = magI1, ipl_lpmagI1 = lpmagI1;
//    IplImage ipl_magI2 = magI2, ipl_lpmagI2 = lpmagI2;
//    cvLogPolar(&ipl_magI1, &ipl_lpmagI1, cvPoint2D32f(magI1.cols >> 1, magI1.rows >> 1), M);
//    cvLogPolar(&ipl_magI2, &ipl_lpmagI2, cvPoint2D32f(magI2.cols >> 1, magI2.rows >> 1), M);
//
//    /// High-Pass Filter Module
//
//    /// Phase correlation Module
//    Mat lpmagI1_64f, lpmagI2_64f;
//    lpmagI1.convertTo(lpmagI1_64f, CV_64F);
//    lpmagI2.convertTo(lpmagI2_64f, CV_64F);
//
//    Point2d pt = phaseCorrelate(lpmagI1_64f, lpmagI2_64f);
//
//    /// scale and rotation from Log-Polar coordinates to angle and scale and output
//    std::cout << "Rotation = " << cv::format("%.2f", pt.y*180/(I1.cols >> 1))
//              << std::endl;
//    std::cout << "Scale = " << exp(pt.x/M) << std::endl;
//    std::cout << "m = " << _m << std::endl;
//
//    /// Transformation Module
//
//    /** Calculate translation */
//    /// Re-scale and de-rotated image
//    Mat rI2 = femtoTrack.rotateImg(I2, center_transform, pt.y*360/(I2.cols), exp(pt.x/M)); // 1/exp(pt.x/m)
//    imshow("rI2", rI2);
//
//    /// Phase correlation
//    Mat rI2_64f;
//    rI2.convertTo(rI2_64f, CV_64FC1);
//    Point2d pt_translation = phaseCorrelate(I1_64f, rI2_64f);
//    cout << "Translation x = " << format("%.3f", pt_translation.x) << endl;
//    cout << "Translation y = " << format("%.3f", pt_translation.y) << endl;
//
//    /** Registered images */
//    /// Translation Module
//
//    /** show the results */
//    /// Transform the matrix with float values into a
//    /// viewable image form (float between values 0 and 1).
//    imshow("Input Image 1"       , I1   );
//    imshow("Input Image 2"       , I2   );
//
//    Mat showMagI1 = Mat::zeros(magI1.size(), magI1.type());
//    Mat showMagI2 = Mat::zeros(magI2.size(), magI2.type());
//    normalize(magI1, showMagI1, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
//    normalize(magI2, showMagI2, 0, 1, CV_MINMAX); // viewable image form (float between values 0 and 1).
//    imshow("magI1", showMagI1);
//    imshow("magI2", showMagI2);
//
//    Mat showLPMagI1 = Mat::zeros(lpmagI1.size(), lpmagI1.type());
//    Mat showLPMagI2 = Mat::zeros(lpmagI2.size(), lpmagI2.type());
//    normalize(lpmagI1, showLPMagI1, 0, 1, CV_MINMAX);
//    normalize(lpmagI2, showLPMagI2, 0, 1, CV_MINMAX);
//    imshow("lpMagI1", showLPMagI1);
//    imshow("lpMagI2Log-Polar 2", showLPMagI2);
//
//    return 0;
//}
