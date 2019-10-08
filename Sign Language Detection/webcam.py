import numpy as np
import cv2
from fastai.vision import * 


# Setting CPU as the default device for inference
defaults.device = torch.device('cpu')


# Main
path = ''
imgpath = 'pred-image.jpg'

# Set the WebCam
cap = cv2.VideoCapture(0)
cap.set(3, 840)
cap.set(4, 680)

# Set Font
font = cv2.FONT_HERSHEY_SIMPLEX

# Set Learner Object
learn = load_learner(path)

# Predict the image
def predict():
    img = open_image(imgpath)
    pred_class, pred_idx, outputs = learn.predict(img)
    return pred_class


def webcam_stream():

    # Initialize fps to 0
    fps = 0

    # Initialize prediction string, title and help text
    prediction = ''
    title = ''
    help_text = 'Press Q to quit and hold S to speak'

    # Run till exit key not pressed - this will capture a video from the webcam
    while True:
        # Capture each frame
        ret, frame = cap.read()
        frame = cv2.flip( frame, 1 )

        if fps == 18: 
            image = frame[50:300, 50:300]
            cv2.imwrite('pred-image.jpg', image)
            pred = predict()
            temp = str(pred)
            if temp == "space":
                prediction += " "
            elif temp == "del":
                prediction = prediction[:-1]
            elif temp == "nothing":
                prediction += ""
            else:
                prediction += temp
            fps = 0

        fps += 1

        # Display Title
        #cv2.putText(frame, title, (180, 30), font,
         #           1, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the prediction underneath the region of interest
        cv2.putText(frame, prediction, (50, 400), font,
                    2, (255, 255, 255), 2, cv2.LINE_AA)

        # Display Help text on the bottom
        cv2.putText(frame, help_text, (50, 650), font,
                    1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw the region of interest and name the video capture window
        cv2.rectangle(frame, (50, 50), (330, 330), (0, 0, 255), 2)
        cv2.imshow("Sign Language Recognition", frame)

        # Exit when q key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Making system say text
        if cv2.waitKey(2) == ord('s'):
            os.system("say '{}'".format(prediction)) 
        

if __name__ == '__main__':
    webcam_stream()


cap.release()
cv2.destroyWindow("Sign Language Recognition")
