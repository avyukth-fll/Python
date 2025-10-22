"""
HSV Filter Tool

This script lets the user select an HSV color range from an image using trackbars,
displays a filtered image that cuts out pixels outside the range, and returns
the selected HSV range.

1. Set the full image path when calling main()
2. Run the code
3. Adjust  trackbars to select a custom HSV range.
4. Press Enter to finish and get the lower and upper HSV bounds that you selected with the taskbars.

Reference:
https://en.wikipedia.org/wiki/Thresholding_(image_processing)
"""

import cv2
import numpy as np
from typing import Tuple


def load_image(path: str) -> np.ndarray:
    """
    Load image from the specified path
    Parameters
    path : str
    Path to the image file.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image at path: {path}")
    return image


def create_hsv_trackbars(window_name: str) -> None:
    """
    Create HSV trackbars for filtering in an OpenCV window.

    Parameters
    window_name : str
    Name of the OpenCV window.
    """
    cv2.createTrackbar("H Lower", window_name, 0, 179, lambda x: None)
    cv2.createTrackbar("S Lower", window_name, 0, 255, lambda x: None)
    cv2.createTrackbar("V Lower", window_name, 0, 255, lambda x: None)
    cv2.createTrackbar("H Upper", window_name, 179, 179, lambda x: None)
    cv2.createTrackbar("S Upper", window_name, 255, 255, lambda x: None)
    cv2.createTrackbar("V Upper", window_name, 255, 255, lambda x: None)


def get_hsv_bounds(window_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    gets hsv bounds from trackbar
    Returns
    tuple[np.ndarray, np.ndarray]
        Lower and upper HSV bounds as NumPy arrays.
    """
    lower_bound = np.array([
        cv2.getTrackbarPos("H Lower", window_name),
        cv2.getTrackbarPos("S Lower", window_name),
        cv2.getTrackbarPos("V Lower", window_name),
    ])
    upper_bound = np.array([
        cv2.getTrackbarPos("H Upper", window_name),
        cv2.getTrackbarPos("S Upper", window_name),
        cv2.getTrackbarPos("V Upper", window_name),
    ])

    if not (lower_bound <= upper_bound).all():
        raise ValueError(f"Lower HSV bound {lower_bound} cannot be greater than upper bound {upper_bound}.")

    return lower_bound, upper_bound


def main(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the HSV Filter tool: show original and filtered images, lets user adjust trackbars,
    and return the selected HSV bounds after enter is pressed
    Parameters
    image_path : str
        Path to the image file to filter.

    Returns
    tuple[np.ndarray, np.ndarray]
        The lower and upper HSV bounds selected by the user.
    """
    #Check if image is valid
    try:
        image = load_image(image_path)
    except FileNotFoundError as e:
        raise e

    window_name = "HSV Filter"
    cv2.namedWindow(window_name)
    create_hsv_trackbars(window_name)

    while True:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        try:
            lower, upper = get_hsv_bounds(window_name)
        except ValueError:
            # Skip frame if trackbar temporarily invalid
            continue

        mask = cv2.inRange(hsv_image, lower, upper)
        filtered_result = cv2.bitwise_and(image, image, mask=mask)

        cv2.imshow("Original", image)
        cv2.imshow("Filtered", filtered_result)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break

    cv2.destroyAllWindows()
    return lower, upper


if __name__ == "__main__":
    # Example usage
    IMAGE_PATH = r"C:\Users\username\New folder\your_image.png"
    lower_bound, upper_bound = main(IMAGE_PATH)
    print(f"HSV Range (Lower): {lower_bound}")
    print(f"HSV Range (Upper): {upper_bound}")
    print(
        f"CSV Format: {lower_bound[0]},{lower_bound[1]},{lower_bound[2]},"
        f"{upper_bound[0]},{upper_bound[1]},{upper_bound[2]}"
    )
