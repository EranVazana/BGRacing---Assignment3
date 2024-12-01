from ultralytics import YOLO
import cv2

# Load input video
input_video_name = 'fsd1.mp4'
video_capture = cv2.VideoCapture(input_video_name)

# Get input video resolution
frame_width = int(video_capture.get(3)) 
frame_height = int(video_capture.get(4)) 
input_video_size = (frame_width, frame_height)

# Load result video recorder
output_video_name = 'Cone_Detection_Video_Result.mp4'
video_recorder = cv2.VideoWriter(output_video_name,  cv2.VideoWriter_fourcc(*'MJPG'),  30, input_video_size) 

# Load pre-trained cone detection model.
model = YOLO('model_trainer\\runs\\detect\\cones\\weights\\best.pt')

while True:
    # Read frame from video.
    retrieve, frame = video_capture.read()
    if (retrieve):
        # Detect and plot racing cones in the retrived frame.
        results = model.predict(frame)[0]
        plotted_frame = results.plot() 
        plotted_frame = cv2.putText(plotted_frame, 'Eran Vazana', (10,35), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,255) , 2, cv2.LINE_AA) 

        # Write the frame into the video recorder.
        video_recorder.write(plotted_frame)

        # Show the frame on the screen.
        cv2.imshow('Cone Detection', plotted_frame)
        
        # Exit Cone Detection if the user pressed the 'q' button
        if (cv2.waitKey(1) == ord('q')):
            break
    else:
        break