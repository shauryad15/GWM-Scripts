
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# Label map
def class_text_to_int(row_label):
    if row_label == 'go':
        return 1
    elif row_label == 'warning':
        return 2
    elif row_label == 'stop':
        return 3
    elif row_label == 'warningLeft':
        return 4
    elif row_label == 'stopLeft':
        return 5



def split(df, group):
    data = namedtuple('data', ['Filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.Filename.split('/')[-1])), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.Filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for index, row in group.object.iterrows():
        if row[1] is not None:
            xmins.append(row[2] / width)
            xmaxs.append(row[4] / width)
            ymins.append(row[3] / height)
            ymaxs.append(row[5] / height)
            classes_text.append(row[1].encode('utf8'))
            classes.append(class_text_to_int(row[1]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter("D:/LISA/lisa-traffic-light-dataset/Annotations/Annotations/frameAnnotationsBOX.record")
    path = os.path.join("D:/LISA/lisa-traffic-light-dataset/daySequence1/daySequence1/frames/")
    examples = pd.read_csv("D:/LISA/lisa-traffic-light-dataset/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv", sep=";")
    grouped = split(examples, 'Filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), "D:/LISA/lisa-traffic-light-dataset/Annotations/Annotations/daySequence1/frameAnnotationsBOX.record")
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
