import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import utils.utils as utils

from config.defaults import cfg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# MOTA metrics imports
import motmetrics as mm

# csv imports
import csv

border_points = []

# set mouse events
def mouse_drawing(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("new border_point")
        print(x, y)

        border_points.append((x,y))
        if len(border_points) > 2:
            border_points.pop(0)

#asdasd
from modeling.yolo import Yolov4
flags.DEFINE_string('weights','data/yolov4-custom_best.weights',
                    'path to weights file')
flags.DEFINE_string('video', './data/video/test2.mpg', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

flags.DEFINE_boolean('track_eval', False, 'enable tracking evaluation')
flags.DEFINE_string('track_eval_gt_path', 'data/dataset/annotations/train/A_d800mm_R1-Filt.csv', 'path to csv GT file from PAMELA dataset')
def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    input_size = cfg.TRAIN.INPUT_SIZE  
    video_path = FLAGS.video
    
    #load model
    model = Yolov4(weight_path=FLAGS.weights, 
               class_name_path=cfg.YOLO.CLASSES)

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    if FLAGS.track_eval:
        name = FLAGS.track_eval_gt_path
        train_rectangles = {}

        with open(name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:  
                frame = int(row[0])
                id =    int(row[1])

                if train_rectangles.get(frame) is None:
                    train_rectangles[frame] = [[], []]
                train_rectangles[frame][0].append(id)
                train_rectangles[frame][1].append([row[3], row[4], row[5], row[6]])


    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    new_objects_positions = {}
    inside_persons_count = 0

    # while video is running
    while True:
        print("_________________________________________________________________________________")
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.setMouseCallback("Output Video", mouse_drawing)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        print("frame: " + str(frame_num))
        frame_num +=1
        #print('Frame #: ', frame_num)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        start_time = time.time()
        preds = model.predict_img(frame)

        bboxes = np.array([[_list[0], _list[1], _list[6], _list[7]] for _list in preds.values.tolist()], dtype=np.float32)
        scores = np.array([_list[5] for _list in preds.values.tolist()], dtype=np.float32)
        num_objects = len(bboxes)
        classes = np.array([0]*num_objects, dtype=np.float32)
        
        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        print(bboxes)
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)


        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        
        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
  
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        old_objects_positions = new_objects_positions
        new_objects_positions = {}

        actual_rectangles = []
        detected_ids = []

        inside_objects = []
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)


        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # save persons positions
            px = ( int((bbox[2] + bbox[0])/2), int((bbox[1] + bbox[3])/2) )
            new_objects_positions[track.track_id] = [px, 0]
        
        # save rectangle
            actual_rectangles.append([int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])])
            detected_ids.append(track.track_id)


        # evaluate 
        if FLAGS.track_eval:     
            if frame_num in train_rectangles:
                print("___ METRICS ___")
                C = mm.distances.iou_matrix(actual_rectangles, train_rectangles[frame_num][1], max_iou=0.5)
                print(C)
                frameid = acc.update( detected_ids, train_rectangles[frame_num][0], C )
                print(acc.mot_events.loc[frameid])

        if FLAGS.count:
            # draw border
            for p in border_points:
                cv2.circle(frame, (p[0], p[1]), 4, (0, 0, 255), -1)

            if len(border_points) == 2:
                # draw border
                cv2.line(frame, border_points[0], border_points[1], (0, 255, 255), 2)

                # draw vector meaning interior
                start_point = (int((border_points[0][0] + border_points[1][0])/2),  
                            int((border_points[0][1] + border_points[1][1])/2))

                end_point =  ((int((start_point[1] - border_points[0][1])/2)) + start_point[0], 
                            (int((start_point[0] - border_points[0][0])/2) * (-1)) + start_point[1])

                cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)

                n = (end_point[0] - start_point[0], end_point[1] - start_point[1])

                for id, coordinate in new_objects_positions.items():
                    v = ( coordinate[0][0] - start_point[0], coordinate[0][1] - start_point[1] )
                    # calculate scalar product 
                    coordinate[1] = np.sign(v[0]*n[0] + v[1]*n[1])
                    print("Tracker ID: {}, Class: {},  Site: {}".format(str(track.track_id), class_name, (coordinate)))
                    if id in old_objects_positions:
                        if (old_objects_positions[id][1] == -1) and coordinate[1] == 1:
                            inside_persons_count = inside_persons_count + 1
                        elif (old_objects_positions[id][1] == 1) and coordinate[1] == -1:
                            inside_persons_count = inside_persons_count - 1
                    
            # print number of person inside
            print("num of person inside: {}".format(inside_persons_count))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'People inside: {inside_persons_count}', (0 + 10, height- 10), font, 0.5, (255,255,0), 1, cv2.LINE_4)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


    # print summary
    print(" ___MOTA___ ")
    mh = mm.metrics.create()

    summary = mh.compute_many(
        [acc, acc.events.loc[0:1]],
        metrics=mm.metrics.motchallenge_metrics,
        names=['full', 'part'],
        generate_overall=True
    )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass