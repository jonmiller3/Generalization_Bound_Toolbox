# Generalization_Bound_Toolbox

Tools related to computing generlization error bounds for machine-learning applications


# Installation

For now, this will not be placed on PyPi. So the following should be performed to manually install. 

Install the build package

    pip install build
    
First, from the same directory as this README, build the sdist and wheel using the following command.

    python -m build

Then install the wheel (*** indicates text that is version and user specific)

    pip install dist/gbtoolbox-***.whl

Build and install the c files

    cd src/gbtoolbox
    make && make install

Update your ld_library_path environment variable

    export LD_LIBRARY_PATH=/usr/local/lib/gbtoolbox/

You may want to add the previous export statement to your ~/.bashrc file, otherwise the change is only for the currently open session. 

# CUDA
There is a CUDA version of a naively implemented DFT function that runs much faster than the C version. The following should be helpful for getting set up to run the CUDA version

    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 cuda-toolkit=11.6 numba python-build scipy -c pytorch -c nvidia
    
# Reference

This toolbox was developed by a collaboration between Euler Scientific ( www.euler-sci.com ) and Fermilab ( www.fnal.gov ). Papers are in progress. Initial developmenet was made possible by the National Geospatial-Intelligence Agency (NGA) under Contract No. HM047622C0003.

The central theory behind this was initially developed by Barron and then extended by E et al.

