# Generalization_Bound_Toolbox

Tools related to computing generlization error bounds for machine-learning applications. Note that standard use depends on the domain of the target functions to be $x \in (-1,1)^d$ where $d$ is the dimension of the feature vectors. If your feature vectors are not in this domain, than they can be rescaled. Additionally, best results are if there is small correlation between any two components of the feature vector.

For directions on use, check out

    tests/test_bound.py

    tests/TestProductSinesCompression.ipynb

    tests/TestProductSines.ipynb.

# Installation

A release version is availalbe on PyPI. Currently requires Python version less than 3.12 and greater than 3.7.

    pip install gbtoolbox

    pip install gbtoolbox[GPU11]

    pip install gbtoolbox[GPU12]    

The following should be performed to manually install. See pyproject.toml for dependencies.

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
There is a legacy CUDA version of nu_dft that runs much faster than the C version, but that runs slower than the cupy version. The following should be helpful for getting set up to run the legacy CUDA version

    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 cuda-toolkit=11.6 numba python-build scipy -c pytorch -c nvidia

    conda install pytorch torchvision torchaudio pytorch-cuda cuda-toolkit numba python-build scipy -c pytorch-nightly -c nvidia

Information about pytorch is available at https://pytorch.org/.

# Reference

This toolbox was developed by a collaboration between Euler Scientific ( www.euler-sci.com ) and Fermilab ( www.fnal.gov ). Papers are in progress. Initial developmenet was made possible by the National Geospatial-Intelligence Agency (NGA) under Contract No. HM047622C0003.

The central theory behind this was initially developed by Barron and then extended by E et al. Details in

https://arxiv.org/abs/1810.06397 

https://arxiv.org/abs/2009.10713

https://arxiv.org/abs/1607.01434

http://www.stat.yale.edu/~arb4/publications_files/UniversalApproximationBoundsForSuperpositionsOfASigmoidalFunction.pdf

