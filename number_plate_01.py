import cv2

# Path to Haar Cascade XML file for license plate detection
harcascade = r"C:\Users\karan\Downloads\haarcascade_russian_plate_number.xml"

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Set frame width and height
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Minimum area for a detected region to be considered as a license plate
min_area = 500

# Counter for saving detected plates
count = 0

while True:
    # Capture a frame from the webcam
    success, img = cap.read()

    # Check if the frame was successfully captured
    if not success:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Create a license plate classifier
    plate_cascade = cv2.CascadeClassifier(harcascade)

    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the grayscale frame
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Iterate through all detected plates
    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            # Draw a rectangle around the detected plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add text "License Plate" above the rectangle
            cv2.putText(img, "License Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            # Extract the region of interest (ROI) containing the license plate
            img_roi = img[y:y + h, x:x + w]
            # Display the ROI in a separate window
            cv2.imshow("ROI", img_roi)

    # Display the frame with rectangles and text
    cv2.imshow("Result", img)

    # Save the detected plate when the 's' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f"license_plate_{count}.jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (10, 250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1

    # Exit the loop when the 'q' key is pressed
    if key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
