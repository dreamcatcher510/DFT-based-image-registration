#include "FemtoTrack.hpp"

using namespace std;
using namespace cv;

FemtoTrack::FemtoTrack()
{
    //ctor
    _m = 40;
}

FemtoTrack::FemtoTrack(int m)
{
    _m = m;
}

FemtoTrack::~FemtoTrack()
{
    //dtor
}

void FemtoTrack::ftTrack1(Mat &a, Mat &b, double &m, double &Tx, double &Ty, double &Rz, double &Scale)
{
//    float m(30.0);   // magnitude scale parameter for log-polar transform
    Mat a_64f, b_64f;
    a.convertTo(a_64f, CV_64FC1);
    b.convertTo(b_64f, CV_64FC1);
    Mat pa = Mat::zeros(a.size(), CV_64FC1); // CV_8UC1
    Mat pb = Mat::zeros(b.size(), CV_64FC1);
    IplImage ipl_a = a_64f, ipl_pa = pa;
    IplImage ipl_b = b_64f, ipl_pb = pb;
    cvLogPolar(&ipl_a, &ipl_pa, cvPoint2D32f(a.cols/2.0, a.rows/2.0), m);
    cvLogPolar(&ipl_b, &ipl_pb, cvPoint2D32f(b.cols/2.0, b.rows/2.0), m);

    Point2d pt = phaseCorrelate(pa, pb);

    Mat br = rotateImg(b, Point2f(a.cols/2.0, a.rows/2.0),-pt.y*360/(a.cols), 1/exp(pt.x/m));
    br.convertTo(br, CV_64FC1);

    Point2d pt1 = phaseCorrelate(a_64f, br);

    Tx = pt1.x;
    Ty = pt1.y;
    Rz = pt.y*360/(a.cols);
    Scale = exp(pt.x/m);
}

void FemtoTrack::ftTrack2(Mat &a, Mat &b, double &th, double &m, double &Tx, double &Ty, double &Rz, double &Scale)
{
    Mat fa = Mat::zeros(a.size(), CV_8UC1);
    Mat fb = Mat::zeros(b.size(), CV_8UC1);
    Canny( a, fa, th, th*3, 3);
    Canny( b, fb, th, th*3, 3);

    fa.convertTo(fa, CV_64FC1);
    fb.convertTo(fb, CV_64FC1);
    Mat pfa = Mat::zeros(fa.size(), CV_64FC1);
    Mat pfb = Mat::zeros(fb.size(), CV_64FC1);
    IplImage ipl_fa = fa, ipl_pfa = pfa;
    IplImage ipl_fb = fb, ipl_pfb = pfb;
    cvLogPolar(&ipl_fa, &ipl_pfa, cvPoint2D32f(fa.cols/2.0, fa.rows/2.0), m);
    cvLogPolar(&ipl_fb, &ipl_pfb, cvPoint2D32f(fb.cols/2.0, fb.rows/2.0), m);

    Point2d pt = phaseCorrelate(pfa, pfb);

    Mat fbr = rotateImg(fb, Point2f(a.cols/2.0, a.rows/2.0), -pt.y*360/(a.cols), 1/exp(pt.x/m));
    fbr.convertTo(fbr, CV_64FC1);

    Point2d pt1 = phaseCorrelate(fa, fbr);

    Tx = pt1.x;
    Ty = pt1.y;
    Rz = pt.y*360/(a.cols);
    Scale = exp(pt.x/m);
}

void FemtoTrack::ftTrack3(Mat &a, Mat &b, double &m, double &Tx, double &Ty, double &Rz, double &Scale)
{
//    float m(30.0);   // magnitude scale parameter for log-polar transform
    Mat ma = magnitudeDFT(a);
    Mat mb = magnitudeDFT(b);
    ma.convertTo(ma, CV_64FC1);
    mb.convertTo(mb, CV_64FC1);
    Mat pma = Mat::zeros(a.size(), CV_64FC1);
    Mat pmb = Mat::zeros(b.size(), CV_64FC1);
    IplImage ipl_ma = ma, ipl_pma = pma;
    IplImage ipl_mb = mb, ipl_pmb = pmb;
    cvLogPolar(&ipl_ma, &ipl_pma, cvPoint2D32f(ma.cols/2.0, ma.rows/2.0), m);
    cvLogPolar(&ipl_mb, &ipl_pmb, cvPoint2D32f(mb.cols/2.0, mb.rows/2.0), m);

    Point2d pt = phaseCorrelate(pma, pmb);

    Mat br = rotateImg(b, Point2f(a.cols/2.0, a.rows/2.0), -pt.y*360/(a.cols), 1/exp(pt.x/m));
    br.convertTo(br, CV_64FC1);
    Mat a_64f;
    a.convertTo(a_64f, CV_64FC1);

    Point2d pt1 = phaseCorrelate(a_64f, br);

    Tx = pt1.x;
    Ty = pt1.y;
    Rz = pt.y*360/(a.cols);
    Scale = exp(pt.x/m);
}


Mat FemtoTrack::magnitudeDFT(Mat& I)
{
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    /**************************************************************/
    /// Switch to a logarithmic scale: for visualization
    /// we can transform our linear scale to a logarithmic one
//    magI += Scalar::all(1);                    // switch to logarithmic scale
//    log(magI, magI);
    /**************************************************************/

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

//    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
//                                            // viewable image form (float between values 0 and 1).
//
//    imshow("Input Image"       , I   );    // Show the result
//    imshow("spectrum magnitude", magI);
//    waitKey();

    return magI;
}

// Log-Polar image
Mat FemtoTrack::logPolar(Mat & I)
{
    if(I.empty())
        std::cout << "The input Mat is empty in the function FemtoTrack::logPolar." <<std::endl;
    I.convertTo(I, CV_64FC1);
    Mat lpI = Mat::zeros(I.size(), CV_64FC1);
    IplImage ipl_I = I, ipl_lpI = lpI;
    cvLogPolar(&ipl_I, &ipl_lpI, cvPoint2D32f(I.cols >> 1, I.rows >> 1), _m);

    return lpI;
}

// input img, centre of rotation, angle or rotation in degrees
Mat FemtoTrack::translateImg(Mat &src, double tx, double ty)
{
    Mat dst = Mat::zeros( src.rows, src.cols, src.type() );
    Mat transMat = (Mat_<double>(2,3) << 1.0, 0, tx, 0, 1.0, ty);
    // warpAffine(src, dst, transMat, dst.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    warpAffine(src, dst, transMat, dst.size(), WARP_INVERSE_MAP, BORDER_CONSTANT, 0);
    return dst;
}


// input img, centre of rotation, angle or rotation in degrees
Mat FemtoTrack::rotateImg(Mat &src, Point2f rotCentre, double rotAngle, double scale)
{
    Mat dst = Mat::zeros( src.rows, src.cols, src.type() );
//    Point2f pt(src.cols/2., src.rows/2.);
    Mat rotMat = getRotationMatrix2D(rotCentre, rotAngle, scale);
    warpAffine(src, dst, rotMat, dst.size(), WARP_INVERSE_MAP, BORDER_CONSTANT, 0); //INTER_LINEAR
    return dst;
}


// compute and show histogram
void FemtoTrack::showHistogram(Mat& img)
{
      int bins = 256;             // number of bins: 8-depth or 8U
      int nc = img.channels();    // number of channels
      vector<Mat> hist(nc);       // array for storing the histograms
      vector<Mat> canvas(nc);     // images for displaying the histogram
      int hmax[3] = {0,0,0};      // peak value for each histogram

      //initialize the hist arrays
      for (int i = 0; i < (int) hist.size(); i++)
            hist[i] = Mat::zeros(1, bins, CV_32SC1);

      //calculation of the number of pixels for each value in each channel
      for (int i = 0; i < img.rows; i++){
            for (int j = 0; j < img.cols; j++){
                  for (int k = 0; k < nc; k++){
                        uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
                        hist[k].at<int>(val) += 1;
                  }
            }
      }

      //compute the max value for every bin
      for (int i = 0; i < nc; i++){
            for (int j = 0; j < bins-1; j++)
                  hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
      }

      //display the histograms
      const char* wname[3] = { "blue", "green", "red" };
      Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };
      for (int i = 0; i < nc; i++){
            canvas[i] = Mat::ones(125, bins, CV_8UC3);
            for (int j = 0, rows = canvas[i].rows; j < bins-1; j++){
                  line(
                  canvas[i],
                  Point(j, rows),
                  Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])),
                  nc == 1 ? Scalar(200,200,200) : colors[i],
                  1, 8, 0);
            }
            imshow(nc == 1 ? "gray" : wname[i], canvas[i]);
            waitKey(0);
      }
}


//compute approximate skeleton
void FemtoTrack::skeleton(Mat& img, Mat& skel)
{
      Mat temp;
      Mat eroded;
      Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
      bool done;
      do{
            erode(img, eroded, element);
            dilate(eroded, temp, element); // temp = open(img)
            subtract(img, temp, temp);
//            morphologyEx(img, eroded, MORPH_GRADIENT, element);
            bitwise_or(skel, temp, skel);
            eroded.copyTo(img);

            done = (countNonZero(img) == 0);
      } while (!done);
}

// compute and show a row
void FemtoTrack::showProfile(Mat& img, int& row)
{
      int bins = img.cols;             // number of bins: 8-depth or 8U
      int nc = img.channels();    // number of channels
      vector<Mat> profil(nc);       // array for storing the profil
      vector<Mat> canvas(nc);     // images for displaying the histogram
      int vmax[3] = {0,0,0};      // peak value for each histogram

      //initialize the profil arrays
      for (int i = 0; i < (int)profil.size(); i++)
            profil[i] = Mat::zeros(1, bins, CV_32SC1);

      //calculation of the value for each pixel in each channel
      for (int j = 0; j < img.cols; j++){
            for (int k = 0; k < nc; k++){
                  profil[k].at<int>(j) = nc == 1 ? img.at<uchar>(Point(row,j)) : img.at<Vec3b>(row,j)[k];
            }
      }

      //compute the max value for every channel
      for (int i = 0; i < nc; i++){
            for (int j = 0; j < bins-1; j++)
                  vmax[i] = profil[i].at<int>(j) > vmax[i] ? profil[i].at<int>(j) : vmax[i];
      }

      //display the profil
      const char* wname[3] = { "blue", "green", "red" };
      Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };
      for (int i = 0; i < nc; i++){
            canvas[i] = Mat::ones(125, bins, CV_8UC3);
            for (int j = 0, rows = canvas[i].rows; j < bins-1; j++){
                  line(
                  canvas[i],
                  Point(j, rows),
                  Point(j, rows - (profil[i].at<int>(j) * rows/vmax[i])),
                  nc == 1 ? Scalar(200,200,200) : colors[i],
                  1, 8, 0);
            }
            imshow(nc == 1 ? "gray" : wname[i], canvas[i]);
            waitKey(0);
      }
}

//copy a row to a txt file
void FemtoTrack::copyRowToFile(Mat& img, int& row, const char* file)
{
      ofstream fichier(file, ios::out | ios::trunc);
      for (int i=0; i<img.cols;i++){
      fichier << (double)img.at<uchar>(Point(i,row)) << endl;
      }
      fichier.close();
}

void FemtoTrack::set_m(int m)
{
     _m = m;
}

int FemtoTrack::read_m()
{
    // std::cout << "_m    :" << _m  << std::endl;
    return _m;
}

// tranShift1
Mat FemtoTrack::translateImg(Mat& I, Point2f offset)
{
    Mat transI = Mat::zeros( I.size(), I.type() );
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offset.x, 0, 1, offset.y);
    warpAffine(I, transI, trans_mat, I.size());
    return transI;
}

// tranShift2
void FemtoTrack::tranShift(Mat& src, Point2f& offset){
	Point2f v1[3] = {
		Point2f(0,0),
		Point2f(src.cols,0),
		Point2f(0,src.rows),
	};
	Point2f v2[3] = {
		Point2f(offset),
		Point2f(offset)+v1[1],
		Point2f(offset)+v1[2]
	};
	warpAffine(
		src,src,
		getAffineTransform(v1,v2),
		src.size(),
		BORDER_REPLICATE
	);
}

Mat FemtoTrack::CannyThreshold(Mat &gray)
{
    Mat cedge = Mat::zeros( gray.size(), gray.type() );
    Mat edge;

    /// Reduce noise with a kernel 3x3
    blur( gray, edge, Size(3,3) );

    /// Run the edge detector on grayscale
    int edgeThresh = 30; // lowThreshold
    int ratio = 3;
    int kernel_size = 3;
    Canny( edge, edge, edgeThresh, edgeThresh*ratio, kernel_size );
    cedge = Scalar::all(0);
    /// Using Canny's output as a mask
    gray.copyTo(cedge, edge);
    return cedge;
}

/*** rewrite the phaseCorrelate module of OpenCV (start)*/
void FemtoTrack::magSpectrums( InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    int depth = src.depth(), cn = src.channels(), type = src.type();
    int rows = src.rows, cols = src.cols;
    int j, k;

    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    if(src.depth() == CV_32F)
        _dst.create( src.rows, src.cols, CV_32FC1 );
    else
        _dst.create( src.rows, src.cols, CV_64FC1 );

    Mat dst = _dst.getMat();
    dst.setTo(0);//Mat elements are not equal to zero by default!

    bool is_1d = (rows == 1 || (cols == 1 && src.isContinuous() && dst.isContinuous()));

    if( is_1d )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataSrc = (const float*)src.data;
        float* dataDst = (float*)dst.data;

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = (float)std::sqrt((double)dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                          (double)dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = (float)std::sqrt((double)dataSrc[j]*dataSrc[j] + (double)dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
    else
    {
        const double* dataSrc = (const double*)src.data;
        double* dataDst = (double*)dst.data;

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = std::sqrt(dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                   dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = std::sqrt(dataSrc[j]*dataSrc[j] + dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
}

void FemtoTrack::divSpectrums( InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB)
{
    Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert( type == srcB.type() && srcA.size() == srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    _dst.create( srcA.rows, srcA.cols, type );
    Mat dst = _dst.getMat();

    bool is_1d = (flags & DFT_ROWS) || (rows == 1 || (cols == 1 &&
             srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if( is_1d && !(flags & DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataA = (const float*)srcA.data;
        const float* dataB = (const float*)srcB.data;
        float* dataC = (float*)dst.data;
        float eps = FLT_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                                       (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] +
                                    (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] -
                                    (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                else
                    for( j = 1; j <= rows - 2; j += 2 )
                    {

                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                                       (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] -
                                    (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] +
                                    (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                if( k == 1 )
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                    double re = (double)(dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1]);
                    double im = (double)(dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j+1] = (float)(im / denom);
                }
            else
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                    double re = (double)(dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1]);
                    double im = (double)(dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j+1] = (float)(im / denom);
                }
        }
    }
    else
    {
        const double* dataA = (const double*)srcA.data;
        const double* dataB = (const double*)srcB.data;
        double* dataC = (double*)dst.data;
        double eps = DBL_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                                       dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] +
                                    dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] -
                                    dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                else
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                                       dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] -
                                    dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] +
                                    dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                if( k == 1 )
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                    double re = dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1];
                    double im = dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1];
                    dataC[j] = re / denom;
                    dataC[j+1] = im / denom;
                }
            else
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                    double re = dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1];
                    double im = dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1];
                    dataC[j] = re / denom;
                    dataC[j+1] = im / denom;
                }
        }
    }

}

void FemtoTrack::fftShift(InputOutputArray _out)
{
    Mat out = _out.getMat();

    if(out.rows == 1 && out.cols == 1)
    {
        // trivially shifted.
        return;
    }

    vector<Mat> planes;
    split(out, planes);

    int xMid = out.cols >> 1;
    int yMid = out.rows >> 1;

    bool is_1d = xMid == 0 || yMid == 0;

    if(is_1d)
    {
        xMid = xMid + yMid;

        for(size_t i = 0; i < planes.size(); i++)
        {
            Mat tmp;
            Mat half0(planes[i], Rect(0, 0, xMid, 1));
            Mat half1(planes[i], Rect(xMid, 0, xMid, 1));

            half0.copyTo(tmp);
            half1.copyTo(half0);
            tmp.copyTo(half1);
        }
    }
    else
    {
        for(size_t i = 0; i < planes.size(); i++)
        {
            // perform quadrant swaps...
            Mat tmp;
            Mat q0(planes[i], Rect(0,    0,    xMid, yMid));
            Mat q1(planes[i], Rect(xMid, 0,    xMid, yMid));
            Mat q2(planes[i], Rect(0,    yMid, xMid, yMid));
            Mat q3(planes[i], Rect(xMid, yMid, xMid, yMid));

            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);

            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);
        }
    }

    merge(planes, out);
}

Point2d FemtoTrack::weightedCentroid(InputArray _src, cv::Point peakLocation, cv::Size weightBoxSize, double* response)
{
    Mat src = _src.getMat();

    int type = src.type();
    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

    int minr = peakLocation.y - (weightBoxSize.height >> 1);
    int maxr = peakLocation.y + (weightBoxSize.height >> 1);
    int minc = peakLocation.x - (weightBoxSize.width  >> 1);
    int maxc = peakLocation.x + (weightBoxSize.width  >> 1);

    Point2d centroid;
    double sumIntensity = 0.0;

    // clamp the values to min and max if needed.
    if(minr < 0)
    {
        minr = 0;
    }

    if(minc < 0)
    {
        minc = 0;
    }

    if(maxr > src.rows - 1)
    {
        maxr = src.rows - 1;
    }

    if(maxc > src.cols - 1)
    {
        maxc = src.cols - 1;
    }

    if(type == CV_32FC1)
    {
        const float* dataIn = (const float*)src.data;
        dataIn += minr*src.cols;
        for(int y = minr; y <= maxr; y++)
        {
            for(int x = minc; x <= maxc; x++)
            {
                centroid.x   += (double)x*dataIn[x];
                centroid.y   += (double)y*dataIn[x];
                sumIntensity += (double)dataIn[x];
            }

            dataIn += src.cols;
        }
    }
    else
    {
        const double* dataIn = (const double*)src.data;
        dataIn += minr*src.cols;
        for(int y = minr; y <= maxr; y++)
        {
            for(int x = minc; x <= maxc; x++)
            {
                centroid.x   += (double)x*dataIn[x];
                centroid.y   += (double)y*dataIn[x];
                sumIntensity += dataIn[x];
            }

            dataIn += src.cols;
        }
    }

    if(response)
        *response = sumIntensity;

    sumIntensity += DBL_EPSILON; // prevent div0 problems...

    centroid.x /= sumIntensity;
    centroid.y /= sumIntensity;

    return centroid;
}

Point2d FemtoTrack::phaseCorrelateRes(InputArray _src1, InputArray _src2, InputArray _window, double* response)
{
    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    Mat window = _window.getMat();

    CV_Assert( src1.type() == src2.type());
    CV_Assert( src1.type() == CV_32FC1 || src1.type() == CV_64FC1 );
    CV_Assert( src1.size == src2.size);

    if(!window.empty())
    {
        CV_Assert( src1.type() == window.type());
        CV_Assert( src1.size == window.size);
    }

    int M = getOptimalDFTSize(src1.rows);
    int N = getOptimalDFTSize(src1.cols);

    Mat padded1, padded2, paddedWin;

    if(M != src1.rows || N != src1.cols)
    {
        copyMakeBorder(src1, padded1, 0, M - src1.rows, 0, N - src1.cols, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(src2, padded2, 0, M - src2.rows, 0, N - src2.cols, BORDER_CONSTANT, Scalar::all(0));

        if(!window.empty())
        {
            copyMakeBorder(window, paddedWin, 0, M - window.rows, 0, N - window.cols, BORDER_CONSTANT, Scalar::all(0));
        }
    }
    else
    {
        padded1 = src1;
        padded2 = src2;
        paddedWin = window;
    }

    Mat FFT1, FFT2, P, Pm, C;

    // perform window multiplication if available
    if(!paddedWin.empty())
    {
        // apply window to both images before proceeding...
        multiply(paddedWin, padded1, padded1);
        multiply(paddedWin, padded2, padded2);
    }

    // execute phase correlation equation
    // Reference: http://en.wikipedia.org/wiki/Phase_correlation
    dft(padded1, FFT1, DFT_REAL_OUTPUT);
    dft(padded2, FFT2, DFT_REAL_OUTPUT);

    mulSpectrums(FFT1, FFT2, P, 0, true);

    magSpectrums(P, Pm);
    divSpectrums(P, Pm, C, 0, false); // FF* / |FF*| (phase correlation equation completed here...)

    idft(C, C); // gives us the nice peak shift location...

    fftShift(C); // shift the energy to the center of the frame.

    // locate the highest peak
    Point peakLoc;
    minMaxLoc(C, NULL, NULL, NULL, &peakLoc);

    // get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
    Point2d t;
    t = weightedCentroid(C, peakLoc, Size(5, 5), response);

    // max response is M*N (not exactly, might be slightly larger due to rounding errors)
    if(response)
        *response /= M*N;

    // adjust shift relative to image center...
    Point2d center((double)padded1.cols / 2.0, (double)padded1.rows / 2.0);

    return (center - t);
}
/*** rewrite the phaseCorrelate module of OpenCV (end)*/
