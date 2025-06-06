"""
NMTT: Nano-micromotor Tracking Tool

13/05/2025
    
@author: Rafael Mestre; r.mestre@soton.ac.uk; 2022
Streamlined by Lauren Sheppard

This code was written to be compatible Python 3.6+, as well
as both Windows, Linux and Mac OS.

"""


import cv2
import numpy as np
import pandas as pd
import os

# Globals for ROI selection
drawing = False
start_point = None
end_point = None
roi_selected = False
initial_pos = None

#Select particle. Draw rectangle around particle of intrest. It works best for a symmetri tight recyange sourrounding the particle.
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, roi_selected, initial_pos
#track initial poition. 
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
     
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        roi_selected = True
        initial_pos = np.array([(start_point[0] + end_point[0]) / 2,
                                (start_point[1] + end_point[1]) / 2], dtype=np.float32)
        print(f"Particle selected at: {initial_pos}")

def main(video_path):
    global start_point, end_point, roi_selected, initial_pos

    # Load the video specified by the path
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first_frame = cap.read()

    if not ret:
        print("Failed to read video.")
        return

    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Select Particle")
    #Select particle of interest with mouse - centre of rectangle is initial position
    cv2.setMouseCallback("Select Particle", draw_rectangle)

    while not roi_selected:
        display = first_frame.copy()
        if start_point and end_point:
            cv2.rectangle(display, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Select Particle", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select Particle")

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_gray = gray_first.copy()
    p0 = initial_pos.reshape(-1, 1, 2)

    frame_idx = 0
    motion_data = []
    cumulative_distance = 0.0
    prev_position = tuple(initial_pos)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Lucas-Kanade method to track the motion of the selected point
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        #set coodinate frame
        if st[0][0] == 1:
            x, y = p1[0][0]
            
            #assign pixels a value in micrometers to measure displacement in μm, 
            #given that 3 micrometers = 42 pixels
            microns_per_pixel = 0.07142857142
            #set dx and dy as the x displacement relative to the initial position
            #chmage ititial_pos[0] to previous_pos[0] for MSD relative to previous point
            dx = (x - initial_pos[0]) * microns_per_pixel
            dy = (y - initial_pos[1]) * microns_per_pixel
            #extract t using the frames per second
            t = frame_idx / fps
            #find mean squared distance as the difference relativ to the origin
            msd = dx**2 + dy**2

            # Step distnace as differce between first and last position
            step_distance = np.linalg.norm(np.array([x, y]) - np.array(prev_position))
            #add steps to find the total difference travells
            cumulative_distance += step_distance
            inst_velocity = step_distance * fps  # instantaneous velocity in micrometers per second

            motion_data.append([t, dx, dy, msd, cumulative_distance, inst_velocity])
            prev_position = (x, y)

            # Draw tracking box
            box_width = abs(end_point[0] - start_point[0])
            box_height = abs(end_point[1] - start_point[1])
            top_left = (int(x - box_width / 2), int(y - box_height / 2))
            bottom_right = (int(x + box_width / 2), int(y + box_height / 2))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            p0 = p1
            old_gray = frame_gray.copy()
        else:
            print("Tracking lost.")
            break

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save to Excel
    df = pd.DataFrame(motion_data, columns=[
        "Time (s)", "Δx (μm)", "Δy (μm)", "MSD (μm²)", "Cumulative Distance (μm)", "Instantaneous Velocity (μm/s)" 
    ])
    output_path = os.path.splitext(video_path)[0] + "_tracking_output.xlsx"
    df.to_excel(output_path, index=False)

    # Average velocity
    total_time = df["Time (s)"].iloc[-1]
    total_distance = df["Cumulative Distance (μm)"].iloc[-1]
    avg_velocity = total_distance / total_time if total_time > 0 else 0
    print(f"\nTracking data saved to: {output_path}")
    print(f"Average Velocity: {avg_velocity:.3f} pixels/second")

if __name__ == "__main__":
    video_path = "/Users/jessica/Downloads/NMTT/CHEM324_DATA/bug/Tepache_1x_glucose/85mA_20250512_tepache_1x_glucose_8.avi" #add your own file path 
    main(video_path)