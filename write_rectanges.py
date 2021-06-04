
import time

from absl import app, flags, logging
from absl.flags import FLAGS
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd




flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video_path', 'data/dataset/video/train/A_d800mm_R2.mpg', 'path to input video')
flags.DEFINE_string('csv_path', 'data/dataset/annotations/train/A_d800mm_R2-Filt.csv', 'path to input csv file')
flags.DEFINE_string('output', './outputs/boxes.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')

def main(_argv):

    try:
        vid = cv2.VideoCapture(int(FLAGS.video_path))
    except:
        vid = cv2.VideoCapture(FLAGS.video_path)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    input_size = FLAGS.size

    frame_data = pd.read_csv(FLAGS.csv_path)
    frame_data.columns = ['frame_nr','person_id','class',
                     'top_x','top_y','width','height']
    frame_num = 0
    print(frame_data)

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)

        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        frames_data_this_frame = frame_data[frame_data['frame_nr'] == frame_num].values.tolist()

        for data_this_frame in frames_data_this_frame:
            person_id = data_this_frame[1]
            top_x = data_this_frame[3]
            top_y = data_this_frame[4]
            box_width = data_this_frame[5]
            box_height = data_this_frame[6]
            

            # draw bbox on screen
            color = colors[int(person_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(top_x), int(top_y)), (int(top_x + box_width), int(top_y + box_height)), color, 2)
            cv2.rectangle(frame, (int(top_x), int(top_y) -30), (int(top_x)+(len('person')+len(str(person_id)))*16,  int(top_y)), color, -1)
            cv2.putText(frame, 'person' + "-" + str(person_id),(int(top_x), int(top_y-10)),0, 0.75, (255,255,255),2)


        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass