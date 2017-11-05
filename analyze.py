from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

# Import urllib
from six.moves import urllib

import numpy as np
import tensorflow as tf

FLAGS = None


import pandas as pd

import tempfile
import urllib

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()

train_file.name = "datasets/training_cut.csv"
test_file.name = "datasets/testing_cut.csv"

# split -l 20 file.txt new
# cut -d, -f1,2,3,7,8,10,11 <transactions.csv > transactions_cut.csv

# df=pd.read_csv('transactions.csv',usecols = [0,1,2,6,7,9])
# d = df.values
# l = pd.read_csv('transactions.csv',usecols = [10])
# labels = l.values
# data = np.float32(d)
# labels = np.array(l,'str')
#
# print("building neuron")

#tensorflow
# x = tf.placeholder(tf.float32,shape=(2972958,6))
# x = data
# w = tf.random_normal([100,2972958],mean=0.0, stddev=1.0, dtype=tf.float32)
# y = tf.nn.softmax(tf.matmul(w,x))
#
#
# with tf.Session() as sess:
#     print(sess.run(y))


def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  # l = pd.read_csv(train_file.name,usecols = [10])
  # labels = l.values
  labels = df_data["rewards_earned"].astype(float)#.apply(lambda x: "rewards_earned" in x).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)

print("***1. Define the columns of the CSV file")
CSV_COLUMNS = [
    "transaction_row_id", "transaction_id", "day", "month", "year",
    "merchant_name", "amount", "zipcode", "country", "rewards_earned",
    "customer_id"]

print("***2. Load training and test files")
df_train = pd.read_csv(train_file.name, names=CSV_COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file.name, names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)


month = tf.feature_column.categorical_column_with_vocabulary_list(
    "month", ["January", "February", "March", "April", "May" , "June" , "July" , "August" , "September" , "October" , "November", "December"])

merchant = tf.feature_column.categorical_column_with_hash_bucket(
    "merchant_name", hash_bucket_size=1000)

rewards_earned = tf.feature_column.numeric_column("rewards_earned")
amount = tf.feature_column.numeric_column("amount")
day_of_month = tf.feature_column.numeric_column("day")
zipcode = tf.feature_column.numeric_column("zipcode")
transaction_id = tf.feature_column.numeric_column("transaction_id")
customer_id = tf.feature_column.numeric_column("customer_id")


base_columns = [
    month, merchant, rewards_earned, amount, zipcode, transaction_id, customer_id, day_of_month
]

model_dir = tempfile.mkdtemp()
m = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns)

print("***3. Train model")
train_steps=20
#usually 2000
m.train(
    input_fn=input_fn(train_file.name, num_epochs=None, shuffle=True),
    steps=train_steps)


results = m.evaluate(
    input_fn=input_fn(test_file.name, num_epochs=1, shuffle=False),
    steps=None)
print("model directory = %s" % model_dir)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

# def maybe_download(train_data, test_data, predict_data):
#   """Maybe downloads training data and returns train and test file names."""
#   if train_data:
#     train_file_name = train_data
#   else:
#     train_file = tempfile.NamedTemporaryFile(delete=False)
#     urllib.request.urlretrieve(
#         "train.csv",
#         train_file.name)
#     train_file_name = train_file.name
#     train_file.close()
#     print("Training data is downloaded to %s" % train_file_name)
#
#   if test_data:
#     test_file_name = test_data
#   else:
#     test_file = tempfile.NamedTemporaryFile(delete=False)
#     urllib.request.urlretrieve(
#         "test.csv", test_file.name)
#     test_file_name = test_file.name
#     test_file.close()
#     print("Test data is downloaded to %s" % test_file_name)
#
#   if predict_data:
#     predict_file_name = predict_data
#   else:
#     predict_file = tempfile.NamedTemporaryFile(delete=False)
#     urllib.request.urlretrieve(
#         "predict.csv",
#         predict_file.name)
#     predict_file_name = predict_file.name
#     predict_file.close()
#     print("Prediction data is downloaded to %s" % predict_file_name)
#
#   return train_file_name, test_file_name, predict_file_name
