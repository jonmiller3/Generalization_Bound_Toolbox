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





