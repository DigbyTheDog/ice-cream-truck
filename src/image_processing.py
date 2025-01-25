import sys
import cv2
import numpy as np
import json

# ----------------------------------------------------------
# TWEAKABLE PARAMETERS
# ----------------------------------------------------------
DEBUG_MODE = True
INITIAL_SCALE = 0.5          # First scale factor for easier contour detection
MIN_PAPER_AREA = 1000        # Minimum contour area to recognize the paper
MIN_DRAW_AREA = 30          # Minimum contour area to recognize parts of the drawing

# Global Canny thresholds (paper detection)
CANNY_THRESH1_GLOBAL = 50
CANNY_THRESH2_GLOBAL = 150

# Canny thresholds inside the paper (drawing detection)
CANNY_THRESH1_PAPER = 30
CANNY_THRESH2_PAPER = 100

# Morphological "closing" kernels to help fill gaps in contours
KERNEL_CLOSE1_SIZE = (10, 10)  # Larger kernel to bridge big gaps
KERNEL_CLOSE2_SIZE = (5, 5)    # Smaller kernel for a second pass

# Final edge cleanup ("shave") parameters
KERNEL_SHAVE_SIZE = (3, 3)     # Size of the erosion kernel
ERODE_ITERATIONS = 3           # How many pixels to shave from the outer edges

# Final output width in pixels
FINAL_WIDTH = 300              # Scale the final cropped image to this width


def union_of_all_contours_preserve_colors_with_cleanup_and_crop(input_path, output_path):
    """
    1) Loads the original image and rescales it by INITIAL_SCALE for contour processing.
    2) Detects the paper contour (largest in the scaled image) and removes background.
    3) Detects & fills *all* inner drawing contours in the scaled image.
    4) Erodes the final mask => removes leftover white fringing.
    5) Reconstructs an alpha channel in the *scaled* domain, then upsamples the mask back
       to original size if needed.
    6) Crops the final image to the bounding box of the subject.
    7) Rescales the final cropped image to FINAL_WIDTH.
    8) Saves as a PNG with transparency.
    9) Displays intermediate steps for debugging.
    """

    # -------------------------
    # STEP 0: LOAD IMAGE
    # -------------------------
    original = cv2.imread(input_path)
    if original is None:
        print(f"Could not read image: {input_path}")
        sys.exit(1)

    orig_h, orig_w = original.shape[:2]

    if DEBUG_MODE:
        # (A) Show the Original
        cv2.imshow("Step A - Original Image", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------
    # STEP 1: RESIZE FOR DETECTION
    # -------------------------
    # Scale the image down for easier/faster/cleaner contour detection
    scaled_w = int(orig_w * INITIAL_SCALE)
    scaled_h = int(orig_h * INITIAL_SCALE)

    # Ensure at least 1 pixel in each dimension
    scaled_w = max(1, scaled_w)
    scaled_h = max(1, scaled_h)

    scaled_img = cv2.resize(original, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # [B] Show Scaled Image (for detection)
    if DEBUG_MODE:
        cv2.imshow("Step B - Scaled Image for Contour Detection", scaled_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # From here on, we do exactly what we did before but in the scaled domain.
    h, w = scaled_img.shape[:2]

    # -------------------------
    # STEP 2: DETECT PAPER IN SCALED IMAGE
    # -------------------------
    gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_global = cv2.Canny(blurred, CANNY_THRESH1_GLOBAL, CANNY_THRESH2_GLOBAL)

    contours, _ = cv2.findContours(edges_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found for paper.")
        sys.exit(1)

    # Filter out small contours that can't be the paper
    paper_candidates = [c for c in contours if cv2.contourArea(c) > MIN_PAPER_AREA]
    if not paper_candidates:
        print("No large paper contour found in scaled image. Adjust MIN_PAPER_AREA.")
        sys.exit(1)

    paper_contour = max(paper_candidates, key=cv2.contourArea)

    mask_paper = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_paper, [paper_contour], -1, 255, thickness=cv2.FILLED)

    paper_only = cv2.bitwise_and(scaled_img, scaled_img, mask=mask_paper)

    # [C] Show the Global Edges & Paper Region
    if DEBUG_MODE:
        cv2.imshow("Step C1 - Global Edges (Paper)", edges_global)
        cv2.imshow("Step C2 - Paper Region (Scaled)", paper_only)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------
    # STEP 3: DETECT DRAWING CONTOURS INSIDE PAPER
    # -------------------------
    gray_paper = cv2.cvtColor(paper_only, cv2.COLOR_BGR2GRAY)
    blurred_paper = cv2.GaussianBlur(gray_paper, (5, 5), 0)
    edges_paper = cv2.Canny(blurred_paper, CANNY_THRESH1_PAPER, CANNY_THRESH2_PAPER)

    edges_paper = cv2.bitwise_and(edges_paper, edges_paper, mask=mask_paper)

    inner_contours, _ = cv2.findContours(edges_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not inner_contours:
        print("No inner contours (drawing) found in scaled image.")
        sys.exit(1)

    # Filter out small "drawing" contours
    inner_contours = [c for c in inner_contours if cv2.contourArea(c) > MIN_DRAW_AREA]
    if not inner_contours:
        print("No valid drawing contours after filtering by area.")
        sys.exit(1)

    # [D] Show Edges Inside Paper
    if DEBUG_MODE:
        cv2.imshow("Step D - Edges Inside Paper (Scaled)", edges_paper)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------
    # STEP 4: FILL ALL CONTOURS
    # -------------------------
    mask_all = np.zeros((h, w), dtype=np.uint8)
    for c in inner_contours:
        cv2.drawContours(mask_all, [c], -1, 255, thickness=cv2.FILLED)

    kernel_close1 = np.ones(KERNEL_CLOSE1_SIZE, np.uint8)
    mask_all_closed = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel_close1)

    union_contours, _ = cv2.findContours(mask_all_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_unified = np.zeros_like(mask_all_closed)
    for uc in union_contours:
        cv2.drawContours(mask_unified, [uc], -1, 255, thickness=cv2.FILLED)

    kernel_close2 = np.ones(KERNEL_CLOSE2_SIZE, np.uint8)
    mask_unified = cv2.morphologyEx(mask_unified, cv2.MORPH_CLOSE, kernel_close2)

    # -------------------------
    # STEP 5: FINAL EDGE CLEANUP
    # -------------------------
    kernel_shave = np.ones(KERNEL_SHAVE_SIZE, np.uint8)
    mask_shaved = cv2.erode(mask_unified, kernel_shave, iterations=ERODE_ITERATIONS)

    # [E] Show Final Mask (Scaled)
    if DEBUG_MODE:
        cv2.imshow("Step E - Final Drawing Mask (Scaled)", mask_shaved)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Now we have a mask for the scaled image. We need to:
    # 1) Expand it back to the original size so we can apply it to the original.
    # 2) Then create BGRA with alpha=0 outside the subject.

    # -------------------------
    # STEP 6: UPSCALE THE MASK TO ORIGINAL SIZE
    # -------------------------
    # Use INTER_NEAREST to avoid introducing partial alpha (as NEAREST keeps crisp edges)
    mask_shaved_up = cv2.resize(mask_shaved, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # [F] Show Upscaled Mask
    if DEBUG_MODE:
        cv2.imshow("Step F - Upscaled Mask to Original Size", mask_shaved_up)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------
    # STEP 7: CREATE BGRA AND APPLY MASK
    # -------------------------
    output_bgra = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)

    # Alpha: 255 where mask_shaved_up=255, else 0
    output_bgra[:, :, 3] = mask_shaved_up
    # Optionally set RGB to black where alpha=0
    output_bgra[mask_shaved_up == 0, :3] = 0

    # [G] Preliminary Transparent BGRA (Original Size)
    if DEBUG_MODE:
        cv2.imshow("Step G - Preliminary Transparent BGRA (Original Size)", output_bgra)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------
    # STEP 8: CROP TO SUBJECT
    # -------------------------
    ys, xs = np.where(mask_shaved_up > 0)
    if len(xs) == 0 or len(ys) == 0:
        print("No nonzero pixels found after mask. Cannot crop.")
        sys.exit(1)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    cropped_bgra = output_bgra[min_y:max_y+1, min_x:max_x+1]

    # [H] Show Cropped
    if DEBUG_MODE:
        cv2.imshow("Step H - Cropped to Subject (Original Scale)", cropped_bgra)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------
    # STEP 9: RESIZE FINAL IMAGE TO FINAL_WIDTH
    # -------------------------
    c_h, c_w = cropped_bgra.shape[:2]
    if c_w != 0:
        scale_factor = FINAL_WIDTH / float(c_w)
    else:
        scale_factor = 1.0

    new_w = FINAL_WIDTH
    new_h = int(c_h * scale_factor)

    final_resized = cv2.resize(cropped_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # [I] Show Final Resized
    if DEBUG_MODE:
        cv2.imshow("Step I - Final Resized Image", final_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------
    # STEP 10: SAVE PNG
    # -------------------------
    cv2.imwrite(output_path, final_resized)
    print(f"Saved transparent PNG to: {output_path}")

    detect_green_circles(output_path, "gumball_locations.json")

    return True


def detect_green_circles(image_path, output_json):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define green color range (adjust as needed)
    lower_green = np.array([35, 50, 50])  # Hue, Saturation, Value
    upper_green = np.array([85, 255, 255])

    # Create a mask for green areas
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply a Gaussian blur to smooth the mask
    blurred_mask = cv2.GaussianBlur(green_mask, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred_mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=5, maxRadius=50
    )

    # Prepare output data
    circle_positions = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            circle_positions.append({"x": int(x), "y": int(y), "radius": int(r)})

            # Optional: Draw the circles on the image for debugging
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)  # Circle outline
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)  # Center point

    # Save the detected circle positions to a JSON file
    with open(output_json, "w") as f:
        json.dump(circle_positions, f)

    # Debugging: Show the results
    if DEBUG_MODE:
        cv2.imshow("Green Mask", green_mask)
        cv2.imshow("Detected Circles", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Detected {len(circle_positions)} green circles. Positions saved to {output_json}.")



def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_image> <output_image>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    union_of_all_contours_preserve_colors_with_cleanup_and_crop(input_file, "isolated_drawing.png")


if __name__ == "__main__":
    main()