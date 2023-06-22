#include <math.h>
#include <stdlib.h>
#include <stdio.h>
void nu_dft(double* x, int N, int D, double* y,  double* f, int M, double* yf)
{
    // Directly computes in the DFT for non-uniformly sampled inputs
    // and non-uniformly sampled frequency vectors
    // Args:
    //      x - input to the function under evaluation, N x D array
    //      N - is the number of samples
    //      D - is the number of dimensions per sample
    //      y - the output of the function given the inputs of x, N x 1 array
    //      f - frequency vectors to sample at, M x D array
    //      M - number of frequency vectors to sample at
    //      yf - output Fourier values, 2*M array to hold interleaved complex output

    const double fac = -2.0*M_PI;

    double* w = (double*)malloc(D*sizeof(double));

    int yfi = 0;

    for (int m = 0; m < M; m++)
    {        
        for (int d = 0; d < D; d++)
        {
            w[d] = fac*f[m*D+d];            
        }
        
        double yc = 0;
        double ys = 0;
        int xi = 0;
        for (int n = 0; n < N; n++)
        {
            double wTxn = 0; // w transpose times x_n
            for (int d = 0; d < D; d++)
            {
                wTxn += w[d]*x[xi++];                
            }

            // the following computes exp(j*wTxn) using a trick
            // double t = tan(wTxn*0.5);
            // double tb = 1.0/(t*t+1.0);
            // double s = 2.0*t*tb;   // sin of wTxn
            // double c = 2.0*tb-1.0; // cos of wTxn
	    // note that the above may not be faster on some systems because tangent
	    // can be dramatically slower for some reason. In that case, use the following
	        double s = sin(wTxn);
	        double c = cos(wTxn);
            yc += y[n]*c;
            ys += y[n]*s;
        }
        yf[yfi++] = yc;
        yf[yfi++] = ys;
    }

    free(w);
}

void nu_dft_e(double* x, int N, int D, double* y,  double* f, int M, double* yf)
{
    // Directly computes in the DFT for non-uniformly sampled inputs
    // and non-uniformly sampled frequency vectors
    // Args:
    //      x - input to the function under evaluation, N x D array
    //      N - is the number of samples
    //      D - is the number of dimensions per sample
    //      y - the output of the function given the inputs of x, N x 1 array
    //      f - frequency vectors to sample at, M x D array
    //      M - number of frequency vectors to sample at
    //      yf - output Fourier values, 2*M array to hold interleaved complex output

    // const double fac = -2.0*M_PI;

    double* w = (double*)malloc(D*sizeof(double));

    int yfi = 0;

    for (int m = 0; m < M; m++)
    {        
        for (int d = 0; d < D; d++)
        {
            w[d] = f[m*D+d];            
        }
        
        double yc = 0;
        double ys = 0;
        int xi = 0;
        for (int n = 0; n < N; n++)
        {
            double wTxn = 0; // w transpose times x_n
            for (int d = 0; d < D; d++)
            {
                wTxn += w[d]*x[xi++];                
            }

            // trick to estimate cos(theta)
            // note that the trick first calculates theta/(2*pi), but here we instead don't multiply f by 2*pi
            double f2 = wTxn-0.25-floor(wTxn+0.25);
            double f3 = 16.0*f2*(fabs(f2)-0.5);
            double c = f3+0.225*f3*(fabs(f3)-1.0);

            // use same trick to estimate sin(theta)=-cos(theta+pi/2)
            //wTxn += 0.25;
            //f2 = wTxn-0.25-floor(wTxn+0.25);
            f2 = wTxn-floor(wTxn+0.5);
            f3 = 16.0*f2*(fabs(f2)-0.5);
            double s = f3+0.225*f3*(fabs(f3)-1.0);
            
            yc += y[n]*c;
            ys += y[n]*s;
        }
        yf[yfi++] = yc;
        yf[yfi++] = ys;
    }

    free(w);
}
