# Setup Requirements for running Demo


The easiest way to run the software used in my project is to Install Anaconda

* Go to https://www.continuum.io/downloads and find correct package. Recommend using Python 3.5, likely 64bit.
    - Run the bash command shown on this page.
    - Let it install to your home directory.
    - Select yes for prepending anaconda to your path.


Now install all the required dependencies from there. The following are steps for setting up the environment:


```bash
# Create the env
conda create --name osgeoenv python=3.5
# Activate the env
source activate osgeoenv

# Install dependencies
conda install -c conda-forge gdal
conda install -c conda-forge -c rios arcsi
conda install -c conda-forge scikit-learn
conda install -c conda-forge scikit-image
conda install -c conda-forge matplotlib
conda install -c conda-forge jupyter
```



Do a quick test to check if the env/bin has been added to the PATH env var.

```bash
# Linux/Unix
$ which gdalinfo
/anaconda/envs/osgeoenv/bin/gdalinfo  # This path may vary depending on Operating System
```

Once the installation is complete, clone the git repo and run jupyter.

```bash
# from your own projects folder
git clone https://github.com/michaelconnor00/satimg

# change dir into the repo
cd satimg  # depending on OS

# Start Jupyter
jupyter notebook # start this from a directory where your files are.

```

## Jupyter Usage

A web page will start in your default browser. Open the `g432_obia.ipynb` notebook from here. Once open, ensure the kernel in the top right is the one you made above (ie - osgeo). If not, go to the kernel menu and change to the correct one.

To use the notebook, click on each cell in sequence click the play button. Any output will be shown below the cell. You can make changes to a cell and replay it, but be aware that the cells below a changed cell do not auto replay.

To exit you can close all the jupyter notebooks, but the server needs to be shutdown as well. Go to the terminal window where you started Jupyter and use ctrl-c  and say y to confirm shutdown.