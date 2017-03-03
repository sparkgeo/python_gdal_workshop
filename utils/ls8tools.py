"""Helper tools for Remote Sensing Workshop, specifically for Landsat 8 datasets.
"""

import os
import sys
import subprocess
import numpy as np
import gdal
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import exposure
from sklearn.cluster import MiniBatchKMeans


plt.ion()


def show(bands, enhance=None, cmap=None):
    """Read as arrays, stack, and show using matplotlib.

    Args:
        bands (GDALDataset): either a list of datasets or a single dataset.

    Returns: None, shows image with matplotlib.
    """
    enhancements = {
        'int': exposure.rescale_intensity, # input and output don't matter.
        'equ': exposure.equalize_hist, # expects uint16 input, output float64
        'adapt': exposure.equalize_adapthist, # expects uint16 input, output float64
        'sig': exposure.adjust_sigmoid, # result is between 0 and 1
        'log': exposure.adjust_log, # result is between 0 and 1
        'gam': exposure.adjust_gamma, # result is between 0 and 1
    }

    run_enhance = []
    if enhance is None:
        run_enhance = [
            exposure.equalize_adapthist,
            exposure.adjust_log,
            exposure.rescale_intensity,
        ]
    else:
        for e in enhance:
            run_enhance.append(enhancements.get(e))

    bands_data = []
    if not isinstance(bands, list):
        if bands.RasterCount == 1:
            # Display single dataset, single band
            int_array = bands.GetRasterBand(1).ReadAsArray()
            bands_data.append(int_array)
        elif bands.RasterCount > 1:
            # Display single dataset, multiple bands
            for i in range(3): # Can only display 3
                int_array = bands.GetRasterBand(i+1).ReadAsArray()
                bands_data.append(int_array)
    else:
        # Display multiple single datasets
        for b in bands:
            int_array = b.GetRasterBand(1).ReadAsArray()
            bands_data.append(int_array)

    for r in run_enhance:
        # print(r)
        bands_data = [r(c) for c in bands_data]
        # print(bands_data[0].dtype)

    if bands_data[0].dtype == np.uint16:
        bands_data = [_convert_to_float(b) for b in bands_data]

    if len(bands_data) == 1:
        if cmap is None:
            plt.imshow(bands_data[0], cmap='Greys')
        else:
            plt.imshow(bands_data[0], cmap=get_colormap(cmap))
    elif len(bands_data) == 3:
        bands_data = np.dstack(b for b in bands_data)
        plt.imshow(bands_data)


def _convert_to_float(uint16_array):
    scalar = 2.0**16 - 1
    return uint16_array.astype(np.float32) / scalar


def get_colormap(num):
    return colors.ListedColormap(
        np.random.rand(num, 3)
    )


def subset_raster(ds, projwin, outname=None):
    """Subset the dataset by the proj window.

    Args:
        ds (GDALDataset): The dataset to subset
        projwin ([float*]): 4 element list of coordinates
        outname (str): output file name

    Returns:
        GDALDataset: The new subsetted dataset
    """
    band_filename = get_band_filename(ds.GetFileList())

    if outname is None:
        outname = gen_output_name(band_filename, 'subset')

    if os.path.exists(outname):
        return gdal.Open(outname)

    command = [
        'gdal_translate', '-projwin',
        str(projwin[0]), str(projwin[1]), str(projwin[2]), str(projwin[3]),
        band_filename, outname
    ]

    subprocess.check_output(command)

    return gdal.Open(outname)


def pansharpen(ms_dss, pan_ds, outname=None):
    """Pansharpen the input ms image.

    Args:
        ms_dss (GDALDatasets): The multispectral dataset
        pan_ds (GDALDataset): The panchromatic dataset
        outname (str): name of the output file

    Returns:
        GDALDataset: the new pansharpened dataset
    """
    pan_file = get_band_filename(pan_ds.GetFileList())

    if outname is None:
        outname = gen_output_name(pan_file, 'pan')

    if os.path.exists(outname):
        return gdal.Open(outname)

    if 'win' in sys.platform:
        gdal_pan_cmd = os.path.join(os.path.dirname(sys.path[1]), 'Scripts', 'gdal_pansharpen.py')
    else:
        gdal_pan_cmd = 'gdal_pansharpen.py'

    command = [
        'python', gdal_pan_cmd, pan_file
    ]

    for ds in ms_dss:
        ms_file = get_band_filename(ds.GetFileList())
        command.append(ms_file)

    command.append(outname)
    subprocess.check_output(command)

    return gdal.Open(outname)


def gen_output_name(input_name, postfix):
    """Generates an output file path with the same name and extension.

    Args:
        input_name (string): path of the input image
        postfix (str): The postfix to add to the output filename

    Returns:
        str: output filename
    """
    image_dir, image_name = os.path.split(input_name)
    root_name, extension = os.path.splitext(image_name)
    new_image_name = '%(path)s_%(postfix)s%(ext)s' % {
        'path': root_name, 'postfix': postfix, 'ext': extension
    }
    return os.path.join(image_dir, new_image_name)


def get_band_filename(files):
    """
    Assumed files is a list of a single band and MTL.txt file.
    """
    for f in files:
        if 'txt' not in os.path.splitext(f)[1].lower():
            return f


def ndvi(red_ds, nir_ds, outname=None):
    """Create a new ndvi image.

    Args:
        red_ds (GDALDataset): the red band dataset
        nir_ds (GDALDataset): the nir band dataset
        outname (str): name and path to the output file

    Returns:
        GDALDataset: the opened dataset for the ndvi.
    """
    if outname is None:
        band_filename = get_band_filename(red_ds.GetFileList())
        band_path, band_filename = os.path.split(band_filename)
        outname = os.path.join(band_path, '%s_ndvi.TIF' % band_filename.split('_')[0])

    if os.path.exists(outname):
        print(outname)
        return gdal.Open(outname)

    # Create an output image using basic settings
    drv = gdal.GetDriverByName('GTiff')
    fout = drv.Create(
        outname,
        red_ds.RasterXSize,
        red_ds.RasterYSize,
        1,
        red_ds.GetRasterBand(1).DataType
    )
    fout.SetGeoTransform(red_ds.GetGeoTransform())
    fout.SetProjection(red_ds.GetProjectionRef())
    fout.SetMetadata(red_ds.GetMetadata())

    # Create NDVI
    red_array = red_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nir_array = nir_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # Tell numpy not to complain about division by 0:
    np.seterr(invalid='ignore')

    ndvi = (nir_array - red_array) / (nir_array + red_array)

    # The ndvi value is in the range -1..1, but we want it to be displayable, so:
    # Make the value positive and scale it back up to the 16-bit range:
    ndvi = (ndvi + 1) * (2**16 - 1)

    # And do the type conversion back:
    ndvi = ndvi.astype(np.uint16)

    # Write the new array
    fout.GetRasterBand(1).WriteArray(ndvi)

    return fout


def kmeans(bands, n_clusters=8, max_iter=10, outname=None):
    """Perform KMeans clustering on input dataset.
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

    Args:
        bands (GDALDataset): either  a list of datasets,
            or a dataset with multiple bands
        outname (str): The string output name and path

    Returns:
        GDALDataset: The opened output dataset
    """

    if outname is None:
        if isinstance(bands, list):
            band_filename = get_band_filename(bands[0].GetFileList())
        else:
            band_filename = get_band_filename(bands.GetFileList())

        band_path, band_filename = os.path.split(band_filename)
        outname = os.path.join(band_path, '%s_kmeans.TIF' % band_filename.split('_')[0])

    if os.path.exists(outname):
        os.remove(outname)

    # Define the classifier
    clf = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        batch_size=10000,
        max_no_improvement=100,
        init_size=2000,
        n_init=10,  # default was 3
        reassignment_ratio=0.05
    )

    # Read data from each band
    test_data = []

    if isinstance(bands, list):
        shape = np.ma.shape(bands[0].GetRasterBand(1).ReadAsArray())
        x_size = bands[0].RasterXSize
        y_size = bands[0].RasterYSize
        out_type = bands[0].GetRasterBand(1).DataType
        geo_trans = bands[0].GetGeoTransform()
        geo_proj = bands[0].GetProjectionRef()
        meta_data = bands[0].GetMetadata()
        for b in bands:
            b_array = b.GetRasterBand(1).ReadAsArray()
            test_data.append(b_array.flatten())
    else:
        shape = np.ma.shape(bands.GetRasterBand(1).ReadAsArray())
        x_size = bands.RasterXSize
        y_size = bands.RasterYSize
        out_type = bands.GetRasterBand(1).DataType
        geo_trans = bands.GetGeoTransform()
        geo_proj = bands.GetProjectionRef()
        meta_data = bands.GetMetadata()
        for band in range(bands.RasterCount):
            b_array = bands.GetRasterBand(band+1).ReadAsArray()
            shape = np.ma.shape(b_array)
            test_data.append(b_array.flatten())

    # Convert to float to prevent sklearn error/warning message
    test_data = np.array(test_data, dtype=np.float32)
    test_data = np.transpose(test_data)

    # Performing K-means classification
    clf.fit(test_data)
    predictedClass = clf.predict(test_data)

    predictedClass = predictedClass + 1 #Add 1 to exclude zeros in output raster
    predictedClass = np.reshape(predictedClass, shape) # Reshape the numpy array to match the original image

    # Create an output raster the same size as the input image
    drv = gdal.GetDriverByName('GTiff')
    fout = drv.Create(
        outname,
        x_size,
        y_size,
        1,
        out_type
    )
    fout.SetGeoTransform(geo_trans)
    fout.SetProjection(geo_proj)
    fout.SetMetadata(meta_data)

    # Write classification to band 1
    fout.GetRasterBand(1).WriteArray(predictedClass)
    fout = None
    return gdal.Open(outname)
