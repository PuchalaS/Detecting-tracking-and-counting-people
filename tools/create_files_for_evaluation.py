
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


from modeling.yolo import Yolov4

flags.DEFINE_string('weights','data/yolov4.weights',
                    'path to weights file')
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
    create_dirs(['/data/dataset/input/detection-results' ,'/data/dataset/input/ground-truth'])

    for _set in ["test"]:
        for video_path in video_paths[_set]:

            try:
                vid = cv2.VideoCapture(int(video_path))
            except:
                vid = cv2.VideoCapture(video_path)

            input_size = FLAGS.size

            file_name = os.path.basename(video_path)
            csv_path = FLAGS.csv_dir + '/' + _set + '/' + os.path.basename(video_path)[:-4] + "-Filt.csv"
            print(csv_path)
            frame_data = pd.read_csv(csv_path)
            frame_data.columns = ['frame_nr','person_id','class',
                            'top_x','top_y','width','height']
            frame_num = 0
            model = Yolov4(weight_path=FLAGS.weights, 
                class_name_path=cfg.YOLO.CLASSES)

            while True:
                
                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                frames_data_this_frame = frame_data[frame_data['frame_nr'] == frame_num].values.tolist()

                eval_file = []
                for data_this_frame in frames_data_this_frame:
                    person_id = data_this_frame[1]
                    top_x = data_this_frame[3]
                    top_y = data_this_frame[4]
                    box_width = data_this_frame[5]
                    box_height = data_this_frame[6]

                    frame_name = str(frame_num)

                    eval_annot = (f"person {top_x} {top_y} {top_x + box_width} {top_y + box_height}")
                    eval_file.append(eval_annot)

                if frames_data_this_frame:
                    preds = model.predict_img(frame)
                    detections = []
                    print(preds)
                    bboxes = [[_list[0],_list[1],_list[2],_list[3],_list[4], _list[5]] for _list in preds.values.tolist()]
                    for box in bboxes:
                        print(box)
                        _l = (f"{box[4]} {box[5]} {box[0]} {box[1]} {box[2]} {box[3]}")
                        detections.append(_l)
                    save_name_gt = (f"data/dataset/input/ground-truth/{file_name[:-4]}_{frame_name}.txt")
                    save_name_detections = (f"data/dataset/input/detection-results/{file_name[:-4]}_{frame_name}.txt")
                    with open(save_name_gt, "w") as f:
                        for line in eval_file:
                            f.write("%s\n" % line)
                    with open(save_name_detections, "w") as f:
                        for line in detections:
                            f.write("%s\n" % line)
            break



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass