# Módulo criado para definir funções auxiliares para os processos
# de enquadramento da CNH

from PIL import Image, ImageChops
import cv2
import numpy as np


DEBUG_MODE = False
''' If True print image will show images. If False then no image is shown'''


def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Filter out further noise using median filters (it is "salt and pepper
# noise" filter type)


def filterOutSaltPepperNoise(edgeImg):
    # Get rid of salt & pepper noise.
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)

# Contour detection after "salt and pepper denoising"


def findSignificantContour(edgeImg):
    contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
    # From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

# Automatically crop the output image to reduce white border left
# from the bg remove process


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    # Bounding box given as a 4-tuple defining the left, upper, right,
    # and lower pixel coordinates.
    # If the image is completely empty, this method returns None.
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def resize_proportionally(image, x_dimension=1000):
    ''' Resize an image to the x_dimension width and the height proportionally
    and returns it'''
    old_x_dimension = image.shape[1]
    old_y_dimension = image.shape[0]
    y_dimension = round(old_y_dimension * x_dimension / old_x_dimension)

    return cv2.resize(image, (x_dimension, y_dimension))


def print_image(image, title='Image', resize=True, shouldWait=False):
    ''' Receives an image and prints on screen.
    If no title is passed the default is Image '''
    if not DEBUG_MODE:
        return 0

    if resize:
        image = resize_proportionally(image, x_dimension=500)
    cv2.imshow(title, image)
    if shouldWait:
        cv2.waitKey()


def document_type(pdf_name):
    # returns the type of the document
    if '_A140' in pdf_name:
        doc_type = 'A140'
    elif '_CA' in pdf_name:
        doc_type = 'CA'
    else:
        doc_type = ''

    return doc_type
