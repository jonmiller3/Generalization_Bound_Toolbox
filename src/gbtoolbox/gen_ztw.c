#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

typedef double ft;

void gen_ztw(int m, int d, int k, ft* zv, ft* wv, ft* tv, ft* w, ft* wftm, ft* wfta, ft* magw)
{
    /*
    m - number of random draws to make
    d - number of dimensions
    k - number of frequency vectors to pull from
    zv - 1-D (size m) array of E theory z values
    wv - mxd array that represent the radial frequency vectors drawn
    tv - size m array of E theory t values
    w - frequency vectors available to draw from    
    wftm - magnitude of weighted Fourier values
    wfta - angle of weighted Fourier values
    magw - magnitudes of the frequency vectors in w 

    */
    srand(time(NULL));
    int cnt = 0;    
    ft rmax = 1.0/(double)RAND_MAX;
    while (cnt < m)
    {
        ft t = rmax*(ft)rand();
        ft z = (ft)(rand()&2)-1.0;
        int w_ind = rand()%k; // TLR % may bias the result somewhat
        ft p = wftm[w_ind]*fabs(cos(magw[w_ind]*t-z*wfta[w_ind]));

        ft chance = rmax*(ft)rand();
        
        if (chance < p)
        {
            zv[cnt] = z;
            tv[cnt] = t;
            // wv[cnt,:] = w[w_ind,:]
            memcpy(wv+cnt*d,w+w_ind*d,sizeof(ft)*d);
            cnt++;
        }
    }
}