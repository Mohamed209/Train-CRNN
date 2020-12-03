import cv2
def display_image(image, window_name_prefix='img', resize_height=False):
    """
    Takes a list of OpenCV images to display.
    Pressing any key will close the windows and resume execution.

    :param image: A single numpy array (image arrya) or a list of OpenCV Images to display.
    :param window_name_prefix: Prefix of the name of the window that display the image.
    :param resize_height: Resizes height for image displaying
    """

    cv2.namedWindow(window_name_prefix, cv2.WINDOW_NORMAL)
    # Assert image is either a list of image of a single image
    assert isinstance(image, (list, np.ndarray)
                      ), "Image is neither a list nor a numpy array"

    # Resize height
    height = 1000

    # If image is a list of images
    if isinstance(image, list):
        # Loop over images list
        for index, img in enumerate(image):
            # Check if image needs to be resized
            if resize_height:
                image = imutils.resize(img, height=height)

            cv2.imshow("{} {}".format(window_name_prefix, index + 1), image)
    # Else image is only a single image
    else:
        if resize_height:
            image = imutils.resize(image, height=height)
        cv2.imshow("{}".format(window_name_prefix), image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()