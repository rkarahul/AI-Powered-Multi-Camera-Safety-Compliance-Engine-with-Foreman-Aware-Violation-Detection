# import cv2
# import numpy as np

# # Red color range in HSV
# lower_red1 = np.array([0, 120, 50])
# upper_red1 = np.array([10, 255, 255])
# lower_red2 = np.array([170, 120, 50])
# upper_red2 = np.array([180, 255, 255])

# # Kernel for dilation
# kernel = np.ones((3, 3), np.uint8)

# # Contour filter params
# min_width = 20
# min_height = 20
# max_width = 120
# max_height = 120
# min_area_ratio = 0.2


# def filter_contours(mask, image, min_w, min_h, max_w, max_h, min_ratio):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     red_pixel_counts = []

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         bounding_area = w * h
#         contour_area = cv2.contourArea(cnt)
#         # print(f"Min width {min_width} Recieved Width {w} Max width {max_w}")
#         # print(f"Min height {min_h} Recieved height {h} Max height {max_h}")
#         if (
#             min_w <= w <= max_w and
#             min_h <= h <= max_h and
#             bounding_area > 0 and
#             (contour_area / bounding_area) >= min_ratio
#         ):
#             # count red pixels inside the bounding box
#             roi_mask = mask[y:y+h, x:x+w]
#             red_pixel_count = cv2.countNonZero(roi_mask)
#             red_pixel_counts.append((x, y, w, h, red_pixel_count))

#             # draw rectangle + pixel count
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(image, str(red_pixel_count), (x, y-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

#     return image, red_pixel_counts


# def process_image(image):
#     """Takes BGR image, returns (image_with_boxes, box_pixel_counts)."""
#     if image is None:
#         return None, []

#     # Convert to HSV
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Create red masks
#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     red_mask = cv2.bitwise_or(mask1, mask2)

#     # Dilation
#     red_mask_dilated = cv2.dilate(red_mask, kernel, iterations=1)

#     # Apply mask
#     red_result = cv2.bitwise_and(image, image, mask=red_mask_dilated)

#     # Filter contours and count red pixels per box
#     image_with_boxes, box_pixel_counts = filter_contours(
#         red_mask_dilated, red_result.copy(),
#         min_width, min_height, max_width, max_height, min_area_ratio
#     )

#     return image_with_boxes, box_pixel_counts


# def cap_check(img):
#     image_with_boxes, box_pixel_counts = process_image(img)
#     # print("Box pixel counts:", box_pixel_counts)#(x, y, w, h, red_pixel_count)
#    # print("len(box_pixel_counts)",len(box_pixel_counts))
#     if len(box_pixel_counts) > 0:
#         return True, image_with_boxes
#     else:
#         return False, image_with_boxes



import cv2
import numpy as np

# Red color range in HSV
lower_red1 = np.array([0, 120, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 50])
upper_red2 = np.array([180, 255, 255])

# Kernel for dilation (slightly larger for better connectivity)
kernel = np.ones((5, 5), np.uint8)

# Contour filter params (lowered min size and area ratio for better detection)
min_width = 15
min_height = 15
max_width = 120
max_height = 120
min_area_ratio = 0.1  # Lowered to handle sparse red pixels in caps


def filter_contours(mask, image, min_w, min_h, max_w, max_h, min_ratio):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_pixel_counts = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_area = w * h
        contour_area = cv2.contourArea(cnt)
        ratio = contour_area / bounding_area if bounding_area > 0 else 0
        
        # Debug print (remove in production if too verbose)
        # print(f"Contour: x={x}, y={y}, w={w}, h={h}, contour_area={contour_area}, bounding_area={bounding_area}, ratio={ratio:.3f}")

        if (
            min_w <= w <= max_w and
            min_h <= h <= max_h and
            bounding_area > 0 and
            ratio >= min_ratio
        ):
            # count red pixels inside the bounding box
            roi_mask = mask[y:y+h, x:x+w]
            red_pixel_count = cv2.countNonZero(roi_mask)
            red_pixel_counts.append((x, y, w, h, red_pixel_count))

            # draw rectangle + pixel count
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, str(red_pixel_count), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            # print(f"Valid contour detected: red pixels = {red_pixel_count}")

    return image, red_pixel_counts


def process_image(image):
    """Takes BGR image, returns (image_with_boxes, box_pixel_counts)."""
    if image is None:
        return None, []

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create red masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological closing to connect nearby red pixels (helps with sparse caps)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Dilation (increased iterations for thicker contours)
    red_mask_dilated = cv2.dilate(red_mask, kernel, iterations=2)

    # Apply mask
    red_result = cv2.bitwise_and(image, image, mask=red_mask_dilated)

    # Filter contours and count red pixels per box
    image_with_boxes, box_pixel_counts = filter_contours(
        red_mask_dilated, red_result.copy(),
        min_width, min_height, max_width, max_height, min_area_ratio
    )

    return image_with_boxes, box_pixel_counts


def cap_check(img):
    image_with_boxes, box_pixel_counts = process_image(img)
    # print("len(box_pixel_counts)", len(box_pixel_counts))
    if len(box_pixel_counts) > 0:
        return True, image_with_boxes
    else:
        return False, image_with_boxes