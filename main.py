import cv2
from simple_facerec import SimpleFacerec

def recognize_a_face():
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return False

    known_face_detected = False  # Variable to keep track of whether a known face has been detected

    while True:
        ret , frame = cap.read()

        # Detect Faces
        detected, name = sfr.detect_known_faces(frame)
        face_locations , face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip( face_locations, face_names ):
            # Adjust the coordinates of the face locations back to the original frame size
            y1 , x1 , y2 , x2 = (int(face_loc[0] / sfr.frame_resizing), int(face_loc[1] / sfr.frame_resizing), 
                                int(face_loc[2] / sfr.frame_resizing), int(face_loc[3] / sfr.frame_resizing))
            cv2.putText(frame, name , (x1, y1 - 10) , cv2.FONT_HERSHEY_DUPLEX , 1 , (0,0,200) , 2)
            cv2.rectangle(frame, (x1,y1) , (x2,y2) , (0,0,200) , 2)
        cv2.imshow("frame" , frame)

        if detected and name != "Unknown":
            print(f"Detected known face: {name}")
            cv2.waitKey(2000)  # Wait for 2 seconds
            known_face_detected = True  # Set the variable to True because a known face has been detected

        key = cv2.waitKey(30)  # Wait for 30 milliseconds
        if key == 27 or known_face_detected:  # Break the loop and close the camera if ESC is pressed or a known face has been detected
            break

    cap.release()
    cv2.destroyAllWindows()

    return known_face_detected

# Call the function
if recognize_a_face():
    print("A known face was detected.")
else:
    print("No known face was detected.")
