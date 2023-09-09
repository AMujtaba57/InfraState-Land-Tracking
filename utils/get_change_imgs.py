import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def get_change_with_ssim(image_current, image_past, threshold=50):
    # Convert images to grayscale
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_BGR2GRAY)
    gray_past = cv2.cvtColor(image_past, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the images
    abs_diff = cv2.absdiff(gray_current, gray_past)

    # Apply thresholding to detect changes
    change_map = cv2.threshold(abs_diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # Calculate SSIM
    ssim_value = compare_ssim(gray_current, gray_past)
    
    # Calculate change percentage based on SSIM
    change_percentage = (1 - ssim_value) * 100

    # Find contours of changes
    contours, _ = cv2.findContours(change_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw change regions on the original images
    change_overlay = image_past.copy()
    cv2.drawContours(change_overlay, contours, -1, (0, 0, 255), 2)
    
    return change_overlay, change_percentage