from PIL import Image
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from keras.models import load_model

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image, lanes):
    # Get image ready for feeding into the model
    small_img = Image.fromarray(image).resize((160, 80))  # Resize using PIL
    small_img = np.array(small_img)

    small_img = small_img[None, :, :, :]

    # Make a prediction with the neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to the list for averaging
    lanes.recent_fit.append(prediction)
    # Only use the last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate the average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Convert the data type of lane_drawn to uint8
    lane_drawn = lane_drawn.astype(np.uint8)

    # Re-size to match the original image
    lane_image = Image.fromarray(lane_drawn).resize((1280, 720))

    lane_image = np.array(lane_image)

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result

if __name__ == '__main__':
    # Load the Keras model
    model = load_model('full_CNN_model.h5')
    # Create a lanes object
    lanes = Lanes()

    # Specify the location of the input video
    input_video_path = 'project_video.mp4'
    # Specify where to save the output video
    output_video_path = 'proj_reg_vid.mp4'

    # Load the input video
    clip1 = VideoFileClip(input_video_path)
    # Create the output video clip
    vid_clip = clip1.fl_image(lambda image: road_lines(image, lanes))
    vid_clip.write_videofile(output_video_path, audio=False)
