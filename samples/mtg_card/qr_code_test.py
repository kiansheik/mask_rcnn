import cv2
from pyzbar import pyzbar

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    # rotation_matrix = cv2.getRotationMatrix2D((frame.shape[0]/2, frame.shape[1]/2), 180, 1)
    # frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
    frame = cv2.flip(frame, -1)  # flip the image vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    # blur = cv2.GaussianBlur(gray, (5, 5), 0) #Gaussian blur
    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1
    )
    adap = cv2.threshold(adap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
        1
    ]  # thresholding
    # equalized = cv2.equalizeHist(adap) #histogram equalization
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # dilated = cv2.dilate(equalized, kernel, iterations=1)
    # closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    qr_codes = pyzbar.decode(adap)
    frame = adap
    # Iterate through the detected QR codes
    for qr_code in qr_codes:
        # Get the bounding box coordinates of the QR code
        (x, y, w, h) = qr_code.rect

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Get the text content of the QR code
        data = qr_code.data.decode("utf-8")

        # Draw the text content of the QR code on the frame
        cv2.putText(
            frame, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    # Show the frame with the bounding boxes and text content
    cv2.imshow("QR Code", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("p"):
        print((x, y, w, h))

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
