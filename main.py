import cv2
import numpy as np


def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size=5):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold=50, high_threshold=150):
    """Applies the Canny edge detector"""
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """Applies an image mask."""
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho=1, theta=np.pi / 180, threshold=50, min_line_len=40, max_line_gap=150):
    """Returns an image with hough lines drawn"""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def draw_lines(img, lines):
    """Draw lines onto the image"""
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)


def lane_detection_pipeline(image):
    """Full pipeline for lane detection"""
    gray = grayscale(image)
    blur = gaussian_blur(gray)
    edges = canny(blur)

    # Define region of interest
    height, width = image.shape[:2]
    region = np.array([[
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]], dtype=np.int32)

    masked_edges = region_of_interest(edges, region)

    # Detect lines
    lines = hough_lines(masked_edges)

    # Draw lines
    if lines is not None:
        draw_lines(image, lines)

    return image


def process_video(video_path):
    """Process video file for lane detection"""
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            result = lane_detection_pipeline(frame)
            cv2.imshow('Lane Detection', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage with a video path
video_path = 'test_video.mp4'  # Replace with the path to your dataset video
process_video(video_path)

