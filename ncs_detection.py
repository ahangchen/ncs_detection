import os, sys
import cv2
import shutil
from mvnc import mvncapi as mvnc
from file_helper import write
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def add_str_on_img(image, total_cnt):
    cv2.putText(image, '%d' % total_cnt, (image.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def preprocess_image(input_image):
    PREPROCESS_DIMS = (300, 300)
    preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
    preprocessed = preprocessed - 127.5
    preprocessed = preprocessed * 0.007843
    preprocessed = preprocessed.astype(np.float16)
    return preprocessed


def predict(image, graph):
    image = preprocess_image(image)
    graph.LoadTensor(image, None)
    (output, _) = graph.GetResult()
    num_valid_boxes = output[0]
    predictions = []
    for box_index in range(num_valid_boxes):
        base_index = 7 + box_index * 7

        if (not np.isfinite(output[base_index]) or
                not np.isfinite(output[base_index + 1]) or
                not np.isfinite(output[base_index + 2]) or
                not np.isfinite(output[base_index + 3]) or
                not np.isfinite(output[base_index + 4]) or
                not np.isfinite(output[base_index + 5]) or
                not np.isfinite(output[base_index + 6])):
            continue

        (h, w) = image.shape[:2]
        x1 = max(0, output[base_index + 3])
        y1 = max(0, output[base_index + 4])
        x2 = min(w, output[base_index + 5])
        y2 = min(h, output[base_index + 6])
        pred_class = int(output[base_index + 1]) + 1
        pred_conf = output[base_index + 2]
        pred_boxpts = (y1, x1, y2, x2)

        prediction = (pred_class, pred_conf, pred_boxpts)
        predictions.append(prediction)

    return predictions


def config_init(dataset_pref):
    os.system('mkdir %s_tmp' % dataset_pref)
    os.system('rm %s_tmp/*' % dataset_pref)


def ncs_prepare():
    print("[INFO] finding NCS devices...")
    devices = mvnc.EnumerateDevices()

    if len(devices) == 0:
        print("[INFO] No devices found. Please plug in a NCS")
        quit()

    print("[INFO] found {} devices. device0 will be used. "
          "opening device0...".format(len(devices)))
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    return device


def ncs_clean(detection_graph, device):
    detection_graph.DeallocateGraph()
    device.CloseDevice()


def graph_prepare(PATH_TO_CKPT, device):
    print("[INFO] loading the graph file into RPi memory...")
    with open(PATH_TO_CKPT, mode="rb") as f:
        graph_in_memory = f.read()

    # load the graph into the NCS
    print("[INFO] allocating the graph on the NCS...")
    detection_graph = device.AllocateGraph(graph_in_memory)
    return detection_graph


def label_prepare(PATH_TO_LABELS, NUM_CLASSES):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def result_process(total_cnt, dataset_pref):
    os.system('rm %s_out/*' % dataset_pref)
    os.system('mkdir %s_out' % dataset_pref)
    for i, image_name in enumerate(sorted(os.listdir(dataset_pref + '_tmp'))):
        shutil.copy(('%s_tmp/' % dataset_pref) + image_name, '%s_out/%08d.jpg' % (dataset_pref, i))
    write('total_cnt.txt', '%s\n' % total_cnt)


def predict_filter(predictions, score_thresh):
    num = 0
    boxes = list()
    scores = list()
    classes = list()
    for (i, pred) in enumerate(predictions):
        (cl, score, box) = pred
        if cl == 21 or cl == 45 or cl == 19 or cl == 76 or cl == 546 or cl == 32:
            if score > score_thresh:
                boxes.append(box)
                scores.append(score)
                classes.append(cl)
                num += 1
    return num, boxes, classes, scores


def count_for_video_ncs(img_dir='r10', start_index=0, end_index=2000):
    dataset_pref = img_dir
    config_init(dataset_pref)
    PATH_TO_CKPT = 'model/ncs_mobilenet_ssd_graph'
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 81
    TEST_IMAGE_PATHS = [os.path.join(img_dir, '%08d.jpg' % i) for i in range(start_index, end_index)]

    device = ncs_prepare()
    detection_graph = graph_prepare(PATH_TO_CKPT, device)
    category_index = label_prepare(PATH_TO_LABELS, NUM_CLASSES)
    
    total_cnt = 0
    for image_path in TEST_IMAGE_PATHS:
        if not os.path.exists(image_path):
            continue
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        predictions = predict(image_np, detection_graph)
        score_thresh = 0.6

        num, valid_boxes, valid_classes, valid_scores = predict_filter(predictions, score_thresh)
        total_cnt += num
        add_str_on_img(image_np, num)
        if num > 0:
            result = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(valid_boxes).reshape(num, 4),
                np.squeeze(valid_classes).astype(np.int32).reshape(num, ),
                np.squeeze(valid_scores).reshape(num, ),
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=score_thresh,
                line_thickness=8)
            cv2.imwrite('%s_tmp/%s' % (dataset_pref, image_path.split('/')[-1]),
                        cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        if num == 0:
            cv2.imwrite('%s_tmp/%s' % (dataset_pref, image_path.split('/')[-1]),
                            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print '%s object: %d' % (image_path, num)


    result_process(total_cnt, dataset_pref)
    ncs_clean(detection_graph, device)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
        start_index = int(sys.argv[2])
        end_index = int(sys.argv[3])
    else:
        img_dir = 'r10'
        start_index = 0
        end_index = 1000
    count_for_video_ncs(img_dir, start_index, end_index)
