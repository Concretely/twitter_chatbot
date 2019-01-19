
import os
import uuid

import sh
import argparse

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO

from tensorflow.core.framework.summary_pb2 import Summary

def ensure_local_file(input_file):
  """
  Ensure the training ratings file is stored locally.
  """
  if input_file.startswith('gs:/'):
    input_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(input_path)
    tmp_input_file = os.path.join(input_path, os.path.basename(input_file))
    sh.gsutil("cp", "-r", input_file, tmp_input_file)
    return tmp_input_file
  else:
    return input_file

def open_file(file_name, mode):
    if file_name.startswith('gs:/'):
        file_stream = file_io.FileIO(file_name, mode=mode)
    else:
        file_stream = open (file_name, mode)
    return file_stream

def open_file_for_string(file_name):
    if file_name.startswith('gs:/'):
        file_stream = StringIO(file_io.FileIO(file_name, mode='r').read())
    else:
        file_stream = file_name
    return file_stream

def save_model(model, file_name):
    model.save(os.path.basename(file_name))
    if file_name.startswith('gs:/'):
        with file_io.FileIO(os.path.basename(file_name), mode='rb') as input_f:
            with file_io.FileIO(file_name, mode='wb') as output_f:
                output_f.write(input_f.read())

def write_hptuning_metric(args, metric):
  """
  Write a summary containing the tuning loss metric, as required by hyperparam tuning.
  """
  summary = Summary(value=[Summary.Value(tag='training/hptuning/metric', simple_value=metric)])

  # for hyperparam tuning, we write a summary log to a directory 'eval' below the job directory
  eval_path = os.path.join(args['output_dir'], 'eval')
  summary_writer = tf.summary.FileWriter(eval_path)

  # Note: adding the summary to the writer is enough for hyperparam tuning.
  # The ml engine system is looking for any summary added with the hyperparam metric tag.
  summary_writer.add_summary(summary)
  summary_writer.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True, help='Model Job Directory')
    parser.add_argument('--input-file-dir', required=True, help='where data files are stored')
    parser.add_argument('--output-file-dir', required=True, help='where data files are stored')
    args, _ = parser.parse_known_args()

    return args
