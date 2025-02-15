import cv2

def capture_image(output_path="captured_image.png"):
    # Open the first available webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        cap.release()
        return False

    # Save the image
    cv2.imwrite(output_path, frame)
    print(f"Image saved to {output_path}")

    # Release the webcam
    cap.release()
    return True

if __name__ == "__main__":
    capture_image()