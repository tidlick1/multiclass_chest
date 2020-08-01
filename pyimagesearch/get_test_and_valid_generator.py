from keras_preprocessing.image import ImageDataGenerator

def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=500, batch_size=32,
                                 seed=1, target_w=256, target_h=256):
    """
    Return generator for validation set and test test set using
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.

    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    # print("getting train and valid generators...")
    # # get generator to sample dataset
    # raw_train_generator = ImageDataGenerator().flow_from_dataframe(
    #     dataframe=train_df,
    #     directory=image_dir,
    #     x_col=x_col,
    #     y_col=y_cols,
    #     class_mode="raw",
    #     batch_size=sample_size,
    #     color_mode='grayscale',
    #     shuffle=True,
    #     target_size=(target_w, target_h))
    #
    # # get data sample
    # batch = raw_train_generator.next()
    # data_sample = batch[0]
    #
    # # use sample to fit mean and std for test set generator
    # image_generator = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True)

    # # fit generator to sample from training data
    # image_generator.fit(data_sample)
    # print('Data Generator mean=%.3f, std=%.3f' % (image_generator.mean, image_generator.std))

    # Add this to output the values to a json file
    # print("[INFO] serializing means...")
    # D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    # f = open(config.DATASET_MEAN, "w")
    # f.write(json.dumps(D))
    # f.close()

    image_generator = ImageDataGenerator(rescale=1.0/255.0)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        color_mode='grayscale',
        seed=seed,
        target_size=(target_w, target_h))

    test_generator = image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=6000,
        color_mode='grayscale',
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h))
    return valid_generator, test_generator