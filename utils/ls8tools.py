import os
import subprocess
import numpy as np
import gdal
from matplotlib import pyplot as plt
from skimage import exposure


plt.ion()


def show_bands(red, green, blue):
    """Read as arrays, stack, and show using matplotlib.

    Args:
        red (GDALRaster): the band show as red.
        green (GDALRaster): the band show as green.
        blue (GDALRaster): the band show as blue.

    Returns: None, shows image with matplotlib.
    """

    # enhancements = {
    #     'sigmoid': exposure.adjust_sigmoid,
    #     'log': exposure.adjust_log,
    #     'intensity': exposure.rescale_intensity,
    #     'gamma': exposure.adjust_gamma,
    #     'equalize': exposure.equalize_hist
    # }

    bands_data = []
    provided_bands = [red, green, blue]
    for b in provided_bands:
        bands_data.append(b.ReadAsArray() * 1.0)

    # print(bands_data[0].max(), bands_data[0].min())
    bands_data = [exposure.adjust_log(c) for c in bands_data]
    bands_data = [exposure.rescale_intensity(c) for c in bands_data]
    bands_data = [exposure.equalize_adapthist(c) for c in bands_data]
    # bands_data = [exposure.equalize_hist(c, nbins=200000) for c in bands_data]

    bands_data = np.dstack(b for b in bands_data)

    plt.imshow(bands_data)


def subset_raster(ds, projwin, outname=None):
    """Subset the dataset by the proj window

    Args:
        ds (GDALDataset): The dataset to subset
        projwin ([float*]): 4 element list of coordinates
        outname (str): output file name

    Returns:
        GDALDataset: The new subsetted dataset
    """
    files = ds.GetFileList()

    if outname is None:
        image_dir, image_name = os.path.split(files[0])
        root_name, extension = os.path.splitext(image_name)
        new_image_name = '%s_subset%s' % (root_name, extension)
        outname = os.path.join(image_dir, new_image_name)

    if os.path.exists(outname):
        return gdal.Open(outname)

    command = [
        'gdal_translate', '-projwin',
        str(projwin[0]), str(projwin[1]), str(projwin[2]), str(projwin[3]),
        files[0], outname
    ]

    subprocess.check_output(command)

    return gdal.Open(outname)
