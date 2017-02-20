# Anaconda Setup


Easiest way to get setup for using Python and the various tools is using [Anaconda](https://www.continuum.io/), which is an open data science platform 

*   Go to https://www.continuum.io/downloads and find correct install package for your operating system. Recommend using Python 3.6, and pick the appropriate architecture (32bit or 64bit).
    - You can ignore the downloading of the cheat sheet, I have included it in the workshop materials. 
    - Install as per the instructions on the download page.
    - Let it install to your home directory (if asked).
    - If asked, choose to install for *Just You* so you don't run into admin issues.
    - Select yes for adding Anaconda to your path.

    ![env_python](images/env_python.PNG)




## Anaconda Configuration

The Anaconda environment is made up of various packages. At times, you may need to install a package, or upgrade a package, or remove a package. All these task are done from the Anaconda prompt, using the `condo` command line interface. For our work, we will need GDAL which is missing from the Anaconda distribution. So let's install it.



First, for Windows users (For Linux or MAC users, open terminal), you should have a program in your start menu called *Anaconda Prompt*. Open it and follow the commands below:

```bash
# List all the installed packages
> conda list

# Install GDAL
> conda install gdal

# It will ask you to confirm the install: ([y]/n) -> type y and enter.

# You should now see it in the lsit
> conda list

# For all the conda commands
> conda --help
```

In the workshop materials (see below for download instructions), there is a `condo` cheat sheet. This will have much more advanced usage examples.



## Download Workshop Materials

Once the installation is complete, download the workshop materials (if you haven't already) and download the sample imagery.

* Go to https://github.com/sparkgeo/python_gdal_workshop
  * There is a green button near the top left `clone or download` (if you are a git user, feel free to clone), click this and select *Download ZIP*
  * Save the ZIP to your home folder `C:\\Users\<username>` and extract it there. 
  * From the Anaconda Prompt, change directories: `> cd python_gdal_workshop`.

* To Download the sample images, we will use Landsat that is hosted by AWS at [landsatonaws.com](https://landsatonaws.com/)

  * To find images on this website, you need to know the path and row for the image you want. 
  * Make a directory for the image files, `> mkdir ls8`, then move into that directory `> cd ls8`.
  * Go to https://landsatonaws.com/L8/048/023/LC80480232014249LGN00
  * For this workshop, we don't need all the bands, so download the following bands to the `C:\\Users\<username>\python_gdal_workshop\ls8\` directory. 
    * LC80480232014249LGN00_B2.TIF - Blue Band
    * LC80480232014249LGN00_B3.TIF - Green Band
    * LC80480232014249LGN00_B4.TIF - Red Band
    * LC80480232014249LGN00_B5.TIF - NIR Band
    * LC80480232014249LGN00_B6.TIF - SWIR-1 Band
    * LC80480232014249LGN00_B8.TIF - Pan Band (Maybe??)
    * LC80480232014249LGN00_MTL.txt

* Move back to the project folder `> cd ../ `.

  ​



## Jupyter Notebooks

Anaconda comes with a local web service for running Python notebooks. A Python notebook is an interface that allows users to write Python in a modular manner and execute different blocks of code independently. This will make more sense once we start using it. Jupiter is a common platform for data science users, which includes remote sensing users. 



From the Anaconda prompt:

```bash
# Start Juypter
> jupyter notebook
```



You should be taken to your default browser and see a list of files in the directory where you started the `jupiter notebook` server. 



Open the `Intro to Python.ipynb` notebook from here.



To use the notebook, click on each cell in sequence and click the play button. Any output will be shown below the cell. You can make changes to a cell and replay it, but be aware that the cells below a changed cell **do not** auto replay.

To exit you can close all the jupyter notebooks, but the server needs to be shutdown as well. Go to the Anaconda Prompt (terminal) window where you started Jupyter and use ctrl-c  and say y to confirm shutdown.



## Reading this Document

If you want to view this document outside of Github, download [Typora](https://typora.io/) which is a very user friendly markdown editor and viewer.