#ifndef DEMODFT_H
#define DEMODFT_H

#include <opencv2/opencv.hpp>

using namespace cv;

class DemoDFT
{
    public:
        DemoDFT();
        DemoDFT(int m);
        virtual ~DemoDFT();

        void set_values(int m);
        int  read_values();

        int demoMagDFT();
        int demoCenRotMagDFT();
        int demoRotMagDFT();
        int demoLogPolar();
        int demoRotMagDFTLogPolar();
        int demoRotMagDFTLogPolarPC();
        int demoMagDFTLogPolarPCSimu();
        int demoMagDFTLogPolarPCImage();
        int demoMagDFTLogPolarPCSimuVideo();
        //int demoMagDFTLogPolarRS();
    protected:
        int _m;     // an example
    private:

};
#endif // DFTDEMO_H
