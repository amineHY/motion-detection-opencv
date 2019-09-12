#####################################################################
# Import Libriries
#####################################################################
import cv2
import numpy as np
import time

#####################################################################
print("\n[INFO] Read frames from the webcam / file\n")
#####################################################################
input_file = 'videos/test_fire_2.mp4'

if isinstance(input_file, str):
    video_source = input_file
    input_file_name = input_file[:-4]
    video = cv2.VideoCapture(video_source)
else:
    video_source = 0
    input_file_name = "videos/webcam"
    video = cv2.VideoCapture(video_source)

time.sleep(2)

if video.isOpened() == False:
    print("[INFO] Unable to read the camera feed")

#####################################################################
# Background Extraction
#####################################################################
# Subtractors
knnSubtractor = cv2.createBackgroundSubtractorKNN(100, 400, True)

# Motion detection parameters
percentage = 0.2  # percent
thresholdCount = 1500
movementText = "Movement is Detected"
textColor = (255, 255, 255)
titleTextPosition = (50, 50)
titleTextSize = 1.2
motionTextPosition = (50, 50)
frameIdx = 0


#####################################################################
# Write video settings: Save video + detection on the disk
#####################################################################
save_output = True
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = None


print("[INFO] Processing...(Press q to stop)")

while(1):
    # Return Value and the current frame
    ret, frame = video.read()
    frameIdx += 1
    print('[INFO] Frame Number: %d' % (frameIdx))

    #  Check if a current frame actually exist
    if not ret:
        break

    if writer is None:
        (frame_height, frame_width) = frame.shape[:2]
        output_file_name = input_file_name + \
            '_motion_detection_{}_{}'.format(frame_height, frame_width)+'.mp4'
        writer = cv2.VideoWriter(
            output_file_name, fourcc, 20.0, (frame_width, frame_height))

        output_motion_file_name = input_file_name + \
            '_motion_{}_{}'.format(frame_height, frame_width)+'.mp4'
        writer_motion = cv2.VideoWriter(
            output_motion_file_name, fourcc, 20.0, (frame_width, frame_height),0)

        #############
        pixel_total = frame_height * frame_width
        thresholdCount = (percentage * pixel_total) / 100

        print('[INFO] frame_height={}, frame_width={}'.format(
            frame_height, frame_width))
        print('[INFO] Number of pixels of the frame: {}'.format(pixel_total))
        print('[INFO] Number of pixels to trigger Detection ({}%) : {}'.format(percentage,
                                                                               thresholdCount))

    print("\n[INFO] Perform Movement Detection: KNN")
    #####################################################################
    tic = time.time()
    knnMask = knnSubtractor.apply(frame)
    toc = time.time()
    knnPixelCount = np.count_nonzero(knnMask)
    knnPixelPercentage = (knnPixelCount*100.0)/pixel_total
    print('[INFO] Processing time (Movement Detection): {0:2.2f} ms'.format(
        (toc-tic)*1000))
    print('[INFO] Percentage of Moving Pixel: {0:2.4f} % ({1:d})'.format(
        knnPixelPercentage, knnPixelCount))

    if (knnPixelCount > thresholdCount) and (frameIdx > 1):
        cv2.putText(frame, movementText, motionTextPosition,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display Results
    #####################################################################
    cv2.imshow('Original', frame)
    cv2.imshow('Movement: KNN', knnMask)

    cv2.moveWindow('Original', 50, 50)
    cv2.moveWindow('Movement: KNN',  frame_width, 50)

    # Record Video
    writer.write(frame) if save_output else 0
    writer_motion.write(knnMask) if save_output else 0

    # if the `q` key was pressed, break from the loop
    #####################################################################
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


video.release()
cv2.destroyAllWindows()
