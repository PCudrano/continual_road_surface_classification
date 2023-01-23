# data loaders

import cv2
import numpy as np


def read_image(path, percentage, original_shape, cropped_shape):
  """
  original_shape[1] * (1 - percentage) must be >= cropped_shape[1]

  """
  img_arr = cv2.imread(path)
  img_arr = cv2.resize(img_arr, dsize=(original_shape[1], original_shape[0]))

  bottom_start = int(original_shape[0] * (1 - percentage))
  img_arr = img_arr[bottom_start - cropped_shape[0]: bottom_start, :]
  img_arr = cv2.resize(img_arr, dsize=(cropped_shape[1], cropped_shape[0]))
  img_arr = np.uint8(img_arr)

  return img_arr


def load_split_data(csv_dir, LABELS):
  """
  csv_dir must contain one .csv file for each key in LABELS named in the same way as the key
  add a row in class_name.csv to add a dataset for the class
  each row must have the form: path/to/dataset/class_name, bottom_cropping
  where bottom_cropping represents the percentage of height to discard from the image starting from the bottom
  """
  x_train, y_train = [], []
  x_valid, y_valid = [], []
  x_test, y_test = [], []

  for class_name in LABELS.keys():
    print(f'Loading {class_name} images...')
    class_index = LABELS[class_name]
    csv_filepath = os.path.join(csv_dir, class_name + '.csv')
    #print(f'csv filepath: {csv_filepath}')

    rows = []
    with open(csv_filepath, 'r') as f:
      lines = csv.reader(f)
      for line in lines:
        rows.append(line)
        #print(line)

    for row in rows:
      dataset_path = os.path.join(BASE, row[0])
      bottom_cropping = float(row[1])
      img_names = sorted(os.listdir(dataset_path))

      train_bound = int(len(img_names) * TRAIN_SPLIT)
      valid_bound = int(len(img_names) * (TRAIN_SPLIT + VALID_SPLIT))
      print(f'Loading {len(img_names)} images from {dataset_path}...')

      for image_index, img_name in enumerate(img_names):
        img_filename = os.path.join(dataset_path, img_name)
        img = read_image(path=img_filename,
                         percentage=bottom_cropping,
                         original_shape=(IMG_H, IMG_W, NUM_CHANNELS),
                         cropped_shape=(CROPPED_H, CROPPED_W, NUM_CHANNELS)
                         )
        if image_index < train_bound:
          x_train.append(np.array(img))
          y_train.append(class_index)

          x_train.append(np.array(adjust_gamma(img)))
          y_train.append(class_index)

          x_train.append(np.array(increase_brightness(img)))
          y_train.append(class_index)

        elif image_index < valid_bound:
          x_valid.append(np.array(img))
          y_valid.append(class_index)
        else:
          x_test.append(np.array(img))
          y_test.append(class_index)

  print('Completed.')
  #x_train, y_train = np.array(x_train), np.array(y_train)
  #x_valid, y_valid = np.array(x_valid), np.array(y_valid)
  #x_test, y_test = np.array(x_test), np.array(y_test)

  return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)