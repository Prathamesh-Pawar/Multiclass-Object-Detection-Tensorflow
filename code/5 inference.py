import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd

#import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

flags = tf.app.flags
flags.DEFINE_string('dir','', 'image_directory')#--dir
flags.DEFINE_string('label_dir','','label_directory')
flags.DEFINE_string('model', '', 'path_to_saved_model')
flags.DEFINE_string('output_dir','','path_to_save_csv')
FLAGS = flags.FLAGS

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# List of the strings that is used to add correct label for each box.
path_ckpt=FLAGS.model+'frozen_inference_graph.pb'
#'C:/Users/PRATHAMESH/Desktop/obj/exported_model/frozen_inference_graph.pb'
PATH_TO_LABELS = FLAGS.label_dir+'label_map.pbtxt'
#'C:/Users/PRATHAMESH/Desktop/obj/data/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
classes=33

PATH_TO_TEST_IMAGES_DIR = pathlib.Path(FLAGS.dir)#'C:/Users/PRATHAMESH/Desktop/test'
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
#TEST_IMAGE_PATHS=TEST_IMAGE_PATHS[20:80]

def load_image_into_numpy_array(image):
	(im_width,im_height)=image.size 
	return np.array(image.get_data()).reshape((im_height,im_width,3)).astype(np.unit8)

detection_graph=tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_ckpt,'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map=label_map_util.load_labelmap(PATH_TO_LABELS)
categories= label_map_util.convert_label_map_to_categories(label_map, max_num_classes=33, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

list_of_preds=[]
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
            image=Image.open(image_path)
            image_np=np.asarray(image)
            image_np_expanded =np.expand_dims(image_np, axis=0)
            image_tensor =detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes =detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detection)=sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            list_of_preds.append((classes.tolist()[0],scores.tolist()[0]))
            print((classes,scores))

def get_labels(classe,score):
    preds=[]
    #list_scores=list(score[0])
    for i in range(0,len(score)):
        if score[i]>0.5:
            preds.append(classe[i])
    #print(preds)
    labels=['apple','bread','bruscitt','cake','carrot','cutlet','fennel_gratin','fillet_fish','fries','green_beans',
       'lasagna_bolognese','meat', 'orange', 'pasta', 'pears','peas','pizza', 'pizzoccheri', 'potatoes', 'pudding',
       'rice','salad','salty_cake','savory_pie', 'scallops','soup', 'spinach', 'squid_stew', 'tangerine', 
       'wet_zucchini', 'yogurt','banana','salmon']

    pred_labels=[]
    for i in preds:
        pred_labels.append(labels[int(i)-1])
        
    pred_labels=set(pred_labels)
    pred_labels=list(pred_labels)
    return pred_labels

def get_images(TEST_IMAGE_PATHS):
    input_images=[]
    for i in TEST_IMAGE_PATHS:
        input_images.append(str(i).split('\\')[-1])
    return input_images

input_images = get_images(TEST_IMAGE_PATHS=TEST_IMAGE_PATHS) 
finals=[]
for i in range(0,len(input_images)):
	print(list_of_preds[i][0])
	print(list_of_preds[i][1])
	dw=get_labels(classe=list_of_preds[i][0],score=list_of_preds[i][1])
	finals.append((input_images[i],dw))
df=pd.DataFrame(columns=['image','labels'])
for i in range(0,len(finals)):
    df.loc[i]=[finals[i][0],finals[i][1]]
df.to_csv(FLAGS.output_dir+'output.csv')

    

