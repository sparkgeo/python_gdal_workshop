"""
Copied from https://github.com/willcadell/geotoolbox
landsat.py
landsat analysis package

this is where all happens!
Will Cadell, 28/11/05
"""

import gdal
# import sys
import pickle
import Numeric
import string
from os import path
from gvclassification import GvClassification
import gview


def image_io(rasterin, out_string):
    """Open raster dataset.

    Args:
        filename (str): absolute or relative path to image.

    Returns:
        GDALRaster: GDAL RAster object.
    """
    

    """
    repeated housekeeping jobs:
    1)generating output filename
    2)open raster dataset
    input:
    1)input raster (string path)
    2)string for output name (".tif")
    output:
    1)outname string
    2)gdal dataset
    Interacts with: automatic_classification, rsri, ndvi, tass_cap, enhance,
    """
    inprefix, insuffix = path.splitext(rasterin)
    outname = inprefix + out_string
    ds = gdal.Open(rasterin)

    return outname, ds


def image_band_count(rasterin):
    """
    """
    ds = gdal.Open(rasterin)
    return ds.RasterCount


def manual_classification(rasterin, class_file):
    """
    1)rasterin (string path)
    Takes the first band from an input raster and a manual classification
    in a pickle file to classifiy the raster and print col,row,value,class
    to screen for use in a loooong table in a database
    2)Pickle File containing a list of classes
    Output:
    1)prints "col, row, id, class" to screen
    Interacts with: classify_to_screen
    """
    f = open(class_file, "r")
    class_list = pickle.load(f)
    f.close

    ds = gdal.Open(rasterin)
    bandarray = ds.GetRasterBand(1).ReadAsArray()
    classify_to_screen(bandarray, class_list)

    print "manual classification complete"
    return


def automatic_classification(rasterin, number, type):
    """
    Takes the first band from an input raster and classifiys it
    using the input parameters of number of bands and classification type
    to print col,row,value,class to screen for use in a loooong table
    in a database
    Inputs:
    1)rasterin (string path)
    2)number of classes
    3)classification method, choices of classifiers are between:
    CLASSIFY_DISCRETE = 0
    CLASSIFY_EQUAL_INTERVAL = 1
    CLASSIFY_QUANTILE = 2
    CLASSIFY_NORM_SD = 3
    Output:
    1)prints "col, row, id, class" to screen
    2)pickle file of class boundaries
    Interacts with: image_io, get_classification, classify_to_screen
    """
    # splits the input file name, then adds "_class.txt" to prep the class list
    # output file name

    outname, ds = image_io(rasterin, '_class.pck')

    # Run classification on the raster to generate class list
    class_list = get_classification(rasterin, number, type)

    # opens or creates a file to write to, writes the class list, closes the
    # file
    f = open(outname, "w")
    pickle.dump(class_list, f)
    f.close

    # Get the single band
    bandarray = ds.GetRasterBand(1).ReadAsArray()
    classify_to_screen(bandarray, class_list)

    print "automatic classification complete"
    return


def classify_to_screen(bandarray, class_list):
    """
    Common classification jobs:
    iterates through every pixel and writes the
    result of the classification to the screen
    input:
    1)the band array
    2)list of classes
    output:
    1)prints results to screen
    Interacts with: manual_classification, automatic_classification,
    get_class_of_value
    """
    rowcnt = 0
    for row in bandarray:
        rowcnt = rowcnt + 1
        colcnt = 0
        for column in row:
            colcnt = colcnt + 1
            cls_no = get_class_of_value(class_list, column)
            print "%i,%i,%i,%i" % (colcnt, rowcnt, column, cls_no)
    return


def txt_to_list(in_file, dtype):
    """
    This turns a csv file into a python list object.
    the file should have no headings and should be of the format:
    col 1) Column Number
    col 2) Row Number
    col 3) Value
    input:
    1) text file of above format
    2) data type of array
    output:
    1)list containing:
        1)count of all entries
        2)column max
        3)row max
        4)list of values for whooole textfile
    interacts with: text_to_image
    """
    f = open(in_file, 'r')

    count = 0
    col_max = 0
    row_max = 0
    val_list = []

    while 1:

        text = f.readline()  # read each line
        txt_line = string.split(text, ',')  # turn each line into a string
        if text == "":  # get out clause
            break
        # append the value of 4th col to list
        val_list.append(int(txt_line[3]))
        col_val = int(txt_line[1])  # record the column and row values
        row_val = int(txt_line[2])

        if col_val > col_max:  # compare to see if these values are the largest
            col_max = col_val  # encountered so far if so then they are stored
        if row_val > row_max:
            row_max = row_val

        count += 1

    f.close()

    return ([count, col_max, row_max, val_list])


def array_to_pan(filename, outname, array, dataType=None):
    """
    turn an array into a panchromatic (single band) Geotiff image
    input:
    1)filename of copycat image
    2)output name
    3)array to turn into image
    4)specific datatype, otherwise will be generated
    from copycat image which may not be suitable
    output:
    1) image in location defined by outname
    interacts with: rsri, ndvi, text_to_image
    """

    ds = gdal.Open(filename)
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize

    if dataType == None:
        dsDataType = ds.GetRasterBand(1).DataType
    else:
        dsDataType = dataType

    drv = gdal.GetDriverByName('GTiff')  # Create an ouput image driver
    fout = drv.Create(outname,  # Create an output image using basic settings
                      x_size,
                      y_size,
                      1,
                      dsDataType)

    # paste the geo data
    fout.SetGeoTransform(ds.GetGeoTransform())
    fout.SetProjection(ds.GetProjectionRef())
    fout.SetMetadata(ds.GetMetadata())

    newband = fout.GetRasterBand(1)  # Get the band of the new image

    newband.WriteArray(array)  # Write the new array to it

    return


def arrays_to_RGB(filename, outname, bandR, bandG, bandB, dataType=None):
    """
    turn an array into a RGB (three band) Geotiff image
    input:
    1)filename of copycat image
    2)output name
    3)array fro red band
    4)array for green band
    5)array for blue band
    6)specific datatype, otherwise will be generated
    from copycat image which may not be suitable
    output:
    1) image in location defined by outname
    interacts with: visuals, tass_cap
    """

    # open input file
    ds = gdal.Open(filename)

    # copy the raster dimensions
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize

    if dataType == None:
        dsDataType = ds.GetRasterBand(1).DataType
    else:
        dsDataType = dataType

    # Create geotiff driver object
    drv = gdal.GetDriverByName('GTiff')

    # Create output images with basic settings
    fout = drv.Create(outname,
                      x_size,
                      y_size,
                      3,
                      dsDataType)

    # paste geo data
    fout.SetGeoTransform(ds.GetGeoTransform())
    fout.SetProjection(ds.GetProjectionRef())
    fout.SetMetadata(ds.GetMetadata())

    # Get bands from true image
    Red = fout.GetRasterBand(1)
    Green = fout.GetRasterBand(2)
    Blue = fout.GetRasterBand(3)

    # Populate image with L7 band values
    Red.WriteArray(bandR)
    Green.WriteArray(bandG)
    Blue.WriteArray(bandB)

    return


def get_landsat_bands(ds, bands='123457'):
    """
    open all landsat bands desired for use, 'bands' is a string which is
    searched through as a code if the band is in the code then the band is
    returned else the band is made = 0 all identifiers are returned to the
    calling statement but only the some coded ones contain anything
    inputs:
    1)gdal dataset
    2)'bands' keyword
    outputs:
    1) up to 6 arrays containing the image bands. 6 variables will be returned,
    but the ones not requested will be empty
    interacts with: ndvi_algorithm, rsri_algorithm, tass_cap_algorithm
    """

    if bands.find('1') > -1:
        b1 = ds.GetRasterBand(1).ReadAsArray()
    else:
        b1 = 0
    if bands.find('2') > -1:
        b2 = ds.GetRasterBand(2).ReadAsArray()
    else:
        b2 = 0
    if bands.find('3') > -1:
        b3 = ds.GetRasterBand(3).ReadAsArray()
    else:
        b3 = 0
    if bands.find('4') > -1:
        b4 = ds.GetRasterBand(4).ReadAsArray()
    else:
        b4 = 0
    if bands.find('5') > -1:
        b5 = ds.GetRasterBand(5).ReadAsArray()
    else:
        b5 = 0
    if bands.find('7') > -1:
        b7 = ds.GetRasterBand(6).ReadAsArray()
    else:
        b7 = 0
    return b1, b2, b3, b4, b5, b7


def text_to_image(in_file, datatype, rastercopy):
    """
    turns a textfile into a geotiff image. Assumes that the input will have
    three locational columns (pixel_ID, Col, Row) and a single value column
    (Value)
    inputs:
    1) the input txt file to process
    2) the datatype as an integer choices are:
         float32 - 6
         int32 - 5
         int16 - 3
    3) copycat image
    outputs:
    1) geotiff image
    interacts with: text_to_list, array_to_pan
    """

    # call txt_to_array function and write results to separate variables
    array_tuple = txt_to_list(in_file, datatype)
    count = array_tuple[0]
    col_max = array_tuple[1]
    row_max = array_tuple[2]
    val_list = array_tuple[3]

    # create an array full of zeros the length of the list generated
    val_line_array = Numeric.zeros((1, count))

    # add the list to the array
    val_line_array = Numeric.add(val_line_array, val_list)

    # resize the array  using the col and rows recorded earlier
    val_squ_array = Numeric.resize(val_line_array, (row_max, col_max))

    # generate outname
    inprefix, insuffix = path.splitext(in_file)
    outname = inprefix + '_img.tif'

    # call function to create output image from array
    array_to_pan(rastercopy, outname, val_squ_array, datatype)

    print "%s generated" % (outname)
    return


def get_classification(raster_filename,
                       num_of_classes=10,
                       class_type=1,
                       band_num=1):
    """
    gets the classification for the automatic classification routine
    inputs:
    1)raster path string
    2)number of classes
    3)classification type, Choices of classifiers are between:
        CLASSIFY_DISCRETE = 0
        CLASSIFY_EQUAL_INTERVAL = 1
        CLASSIFY_QUANTILE = 2
        CLASSIFY_NORM_SD = 3
    4)band number to process
    outputs: list of classes
    interacts with:automatic_classification
    """

    ds = gdal.Open(raster_filename)

    # Get the single band
    # band = ds.GetRasterBand(band_num)

    # Create a gview raster object from band
    raster = gview.GvRaster(None, ds)

    # Define gview layer from raster
    # Needed to pass to GvClassification
    layer = gview.GvRasterLayer(raster)

    classification = GvClassification(layer)
    # classification.set_classify_property(layer,str(class_type))
    classification.set_type(class_type)

    # use default classification with number of classes
    # defined at input
    classification.prepare_default(num_of_classes)

    # list for holding the classes
    cls = []

    # cycle through the classes and add them to the cls list
    for cls_in in range(num_of_classes):
        cls.append(classification.get_range(cls_in))

    return cls


def get_class_of_value(class_list, pixel_val):
    """
    find the class of a specific pixel using a list of classes
    input:
    1)list of classes
    2)pixel value
    output:
    1)class value of pixel
    interacts with: classify_to_screen
    """
    # counter variable
    class_num = int(-1)
    y = int(0)

    # compare the value supplied with the classes to find out in which class
    # the variable should sit
    for x in class_list:
        if (pixel_val >= class_list[y][0]) and (pixel_val < class_list[y][1]):
            class_num = y
        y = y + 1

    return class_num


def get_max_min(dataset, band_no):
    """
    takes an open dataset and band number and returns
    the max and min values of that band
    input:
    1)gdal dataset
    2)band number
    output:
    1)minimum value
    2)maximum value
    interacts with: rsri
    """
    band = dataset.GetRasterBand(band_no)
    min = band.ComputeRasterMinMax()[0]
    max = band.ComputeRasterMinMax()[1]

    return min, max


def rsri(rasterin):
    """
    reduced simple ratio index
    input:
    1)raster path string
    output:
    1)image file
    interacts with: image_io, rsri_algorithm, array_to_pan
    """
    outname, ds = image_io(rasterin, '_rsri.tif')
    rsri = rsri_algorithm(ds)
    array_to_pan(rasterin, outname, rsri, 3)

    print "%s generated" % (outname)
    return outname


def rsri_algorithm(ds):
    """
    the rsri algorithm
    inputs
    1) gdal dataset
    outputs:
    2) array of calculated values
    interacts with: rsri, get_min_max, get_landsat_bands
    """
    mir_min, mir_max = get_max_min(ds, 5)
    b1, b2, b3, b4, b5, b7 = get_landsat_bands(ds, '345')
    red = b3 + 0.01
    nir = b4 + 0.01
    mir = b5 + 0.01

    rsri = (((nir / red) * ((mir_max - mir) / (mir_max - 10))) / 4) * 255

    return rsri


def ndvi(rasterin):
    """
    normalised difference index
    input:
    1)raster path string
    output:
    1)image file
    interacts with: image_io, ndvi_algorithm, array_to_pan
    """
    outname, ds = image_io(rasterin, '_ndvi.tif')
    ndvi = ndvi_algorithm(ds)
    array_to_pan(rasterin, outname, ndvi, 3)

    print "%s generated" % (outname)
    return outname


def ndvi_algorithm(ds):
    """
    the ndvi algorithm
    inputs
    1) gdal dataset
    outputs:
    2) array of calculated values
    interacts with: ndvi, get_landsat_bands
    """
    b1, b2, b3, b4, b5, b7 = get_landsat_bands(ds, '34')
    red = b3 + 0.01
    ir = b4 + 0.01

    ndvi = ((ir - red) / (ir + red)) * 255
    ndvi = (ndvi >= 0) * ndvi

    return ndvi


def log(array):
    """
    logarithmic enhancement algorithm using numeric array processing
    input: array
    output: processed array
    interacts with: enhance
    """
    array = (255 * (Numeric.log(1.0 + array) / Numeric.log(256.0))) + 0.5

    return array


def square(array):
    """
    Square enhancement algorithm using numeric array processing.

    input: array
    output: processed array
    interacts with: enhance
    """
    array = 255 * Numeric.power(array / 255.0, 2)

    return array


def root(array):
    """
    root enhancement algorithm using numeric array processing
    input: array
    output: processed array
    interacts with: enhance
    """
    array = 255.0 * Numeric.sqrt(array / 255.0)

    return array


def block_generator(band, start, length):
    """
    This function helps to split a large array into pieces so fast numeric
    processing can be carried out on chunks of data rather than single
    elements (v. slow), lines (see stretch.py, functional) or the whole
    thing (Landsat scene - memory error). This function is the be called
    inside an iteration where 'start' is changing, with each call, by 'length'
    Input: Data Band, Start point of the block, Length of the block
    Output: Array - block of data to be processed
    """
    w_block = list([])  # empty 1d array
    if (length + start) > band.YSize:  # conditional to stop last blk overflow
        length = (band.YSize - start)  # last blk length of last piece of image
    for j in range(length):  # take specific line from image & append 2 list
        w_block.append(band.ReadAsArray(0, (start + j), band.XSize, 1))

    block = Numeric.resize(w_block, (length, band.XSize))  # resize array to 2d

    return block


def enhance(rasterin, alg='root'):
    """
    enhance.py
    Will Cadell, 23/11/05, algorithms adapted from Mario Bauchamps
    histogram enhance toolmore OO version of stretch.py, processing
    blocks not lines
    args: input image, enhancement(root, log or square)
    output: output image as tif
    """

    alg_dict = {  # dictionary to hold references to alogrithms
        "root": root,
        "square": square,
        "log": log,
    }

    out_suffix = alg + '.tif'
    outname, ds = image_io(rasterin, out_suffix)
    band_count = ds.RasterCount  # define the number of bands from dataset

    drv = gdal.GetDriverByName('GTiff')  # Create geotiff driver object
    fout = drv.Create(outname,  # Create output image driver
                      ds.RasterXSize,  # Array Xsize same as input
                      ds.RasterYSize,  # Array Ysize same as input
                      band_count,  # bands same as input
                      3)  # output datatype set to int16

    fout.SetGeoTransform(ds.GetGeoTransform())  # Copy geotransform
    fout.SetMetadata(ds.GetMetadata())  # Copy metadata
    fout.SetProjection(ds.GetProjectionRef())  # Copy projection

    for x in range(band_count):  # iterate through the bands
        band = ds.GetRasterBand(x + 1)  # open the input band
        band_out = fout.GetRasterBand(x + 1)  # open the output band
        for i in range(0, ds.RasterYSize, 100):  # iterate through each block
            block = block_generator(band, i, 100)  # call block generator
            block = alg_dict[alg](block)  # call algorithm dict fr alg
            band_out.WriteArray(block, 0, i)  # write block to new band
        print "band %i of %i complete" % (x + 1, band_count)

    print "Image Complete: %s " % (fout.GetDescription())
    return outname


def visuals(rasterin):
    """
    Produce true and false colour images.

    input: raster sting path
    output: 2 geotiff images
    interacts with: get_landsat_bands, arrays_to_RGB
    """
    inprefix, insuffix = path.splitext(rasterin)
    outname_f = inprefix + '_false.tif'
    outname_t = inprefix + '_true.tif'

    ds = gdal.Open(rasterin)  # Open input image

    b1, b2, b3, b4, b5, b7 = get_landsat_bands(ds, '1234')

    arrays_to_RGB(rasterin, outname_t, b3, b2, b1)
    arrays_to_RGB(rasterin, outname_f, b4, b2, b1)

    print "%s and %s generated" % (outname_t, outname_f)
    return outname_t, outname_f


def tass_cap(rasterin):
    """
    Tasselled cap transform.

    input:
    1)raster path string
    output:
    1)image file
    interacts with: image_io, tass_cap_algorithm, array_to_RGB
    """
    outname, ds = image_io(rasterin, '_tc.tif')
    b, g, w = tass_cap_algorithm(ds)
    arrays_to_RGB(rasterin, outname, b, g, w, 3)

    print "%s generated" % (outname)
    return outname


def tass_cap_algorithm(ds):
    """
    The tasslled cap transfrom algorithm inputs.

    1) gdal dataset
    outputs:
    2) 3 array of red, green, blue
    interacts with: ndvi, get_landsat_bands
    """
    # call landsat band separation script
    b1, b2, b3, b4, b5, b7 = get_landsat_bands(ds)

    # Run tasseled-cap transform
    brightness = ((0.3037 * b1) + (0.2793 * b2) + (0.4343 * b3) +
                  (0.5585 * b4) + (0.5082 * b5) + (0.1863 * b7))
    greeness = ((-0.2848 * b1) + (-0.2435 * b2) + (-0.5436 * b3) +
                (0.7243 * b4) + (0.0840 * b5) + (-0.1800 * b7))
    wetness = ((0.1509 * b1) + (0.1793 * b2) + (0.3299 * b3) +
               (0.3406 * b4) + (-0.7112 * b5) + (-0.4572 * b7))

    return brightness, greeness, wetness
