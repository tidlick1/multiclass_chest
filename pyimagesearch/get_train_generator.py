from keras_preprocessing.image import ImageDataGenerator

def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=32, seed=1, target_w=256, target_h=256):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
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
        train_generator (DataFrameIterator): iterator over training set
        :param df:
        :param image_dir:
        :param x_col:
        :param y_cols:
        :param shuffle:
        :param batch_size:
        :param seed:
        :param target_w:
        :param target_h:
        :return:
    """
    print("getting train generator...")
    # normalize images
    # image_generator = ImageDataGenerator(
    #     samplewise_center=True,
    #     samplewise_std_normalization=True)

    image_generator = ImageDataGenerator(rescale=1.0/255.0)

    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=shuffle,
        seed=seed,
        target_size=(target_w, target_h))

    return generator