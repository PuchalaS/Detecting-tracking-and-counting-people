
import os 
import sys
sys.path.append('.')
from absl import app, flags, logging
from absl.flags import FLAGS
from config.defaults import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import create_dirs
import pandas as pd



flags.DEFINE_string('output_images_dir', 'data/dataset/images', 'path to output video')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video_dir', 'data/dataset/video', 'path to directory containing train and test video folders')
flags.DEFINE_string('csv_dir', 'data/dataset/annotations', 'path to directory containing train and test annotations folders')
flags.DEFINE_string('output_dir', './outputs', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('save_GT_videos', False, 'save video with boxes from ground truth files')


def main(_argv):
    video_paths = {}
    for root, dirs, files in os.walk(FLAGS.video_dir):
        print(f'{root}, {dirs}, {files}')
        if files:
            video_paths[os.path.basename(root)] = [root +'/'+ file_ for file_ in files]
    #annot_files = 
    print(video_paths)
    create_dirs([FLAGS.output_images_dir + '/train' , FLAGS.output_images_dir + '/test'])

    processed_csv_data = []
    for _set in ["test", "train"]:
        for video_path in video_paths[_set]:

            try:
                vid = cv2.VideoCapture(int(video_path))
            except:
                vid = cv2.VideoCapture(video_path)

            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(FLAGS.output_dir +'/'+os.path.basename(video_path)[:-4] + "-GT.avi", codec, fps, (width, height))
            input_size = FLAGS.size

            file_name = os.path.basename(video_path)
            csv_path = FLAGS.csv_dir + '/' + _set + '/' + os.path.basename(video_path)[:-4] + "-Filt.csv"
            print(csv_path)
            frame_data = pd.read_csv(csv_path)
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


                image_data = cv2.resize(frame, (input_size, input_size))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)

                #initialize color map
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
                save_name_annot = ''
                yolo_annot =''
                frames_data_this_frame = frame_data[frame_data['frame_nr'] == frame_num].values.tolist()
                txt_file = []
                for dx, data_this_frame in enumerate(frames_data_this_frame):
                    person_id = data_this_frame[1]
                    top_x = data_this_frame[3]
                    top_y = data_this_frame[4]
                    box_width = data_this_frame[5]
                    box_height = data_this_frame[6]

                    center_x = top_x + int(box_width/2)
                    center_y = top_y + int(box_height/2)

                    #save frame to .jpg for traning purposes

                    if len(str(frame_num)) == 3:
                        frame_name = "0" + str(frame_num)
                    else:
                        frame_name = str(frame_num)

                    save_name_jpg = (f"{FLAGS.output_images_dir}/{_set}/{file_name[:-4]}_{frame_name}.jpg")
                    save_name_annot = (f"{FLAGS.output_images_dir}/{_set}/{file_name[:-4]}_{frame_name}.txt")
                    if dx == 0:
                        cv2.imwrite(save_name_jpg, frame)
                    yolo_annot = (f"0 {center_x/width} {center_y/height} {box_width/width} {box_height/height}")
                    txt_file.append(yolo_annot)


                    # draw bbox on screen
                    color = colors[int(person_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(top_x), int(top_y)), (int(top_x + box_width), int(top_y + box_height)), color, 2)
                    cv2.rectangle(frame, (int(top_x), int(top_y) -30), (int(top_x)+(len('person')+len(str(person_id)))*16,  int(top_y)), color, -1)
                    cv2.putText(frame, 'person' + "-" + str(person_id),(int(top_x), int(top_y-10)),0, 0.75, (255,255,255),2)


                if save_name_annot:
                    with open(save_name_annot, "w") as f:
                        for line in txt_file:
                            f.write("%s\n" % line)

                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if not FLAGS.dont_show:
                    cv2.imshow("Output Video", result)

                # if output flag is set, save video file
                if FLAGS.save_GT_videos:
                    out.write(result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break


            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass