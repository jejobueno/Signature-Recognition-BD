import pandas as pd


def x_center(df):
    return int(df.x_scaled + (df.w_scaled / 2))


def y_center(df):
    return int(df.y_scaled + (df.h_scaled / 2))


def w_norm(df, col):
    return df[col] / df['page_width_scaled']


def h_norm(df, col):
    return df[col] / df['page_height_scaled']


def transform_to_YOLOv5_form(df: pd.DataFrame):
    """
    :param df: Data frame with coordinates of the signatures in the different scaled images
    :return: df with YOLOv5 formatted coordinates
    """

    df['x_center'] = df.apply(x_center, axis=1)
    df['y_center'] = df.apply(y_center, axis=1)

    df['x_center_norm'] = df.apply(w_norm, col='x_center', axis=1)
    df['width_norm'] = df.apply(w_norm, col='w_scaled', axis=1)

    df['y_center_norm'] = df.apply(h_norm, col='y_center', axis=1)
    df['height_norm'] = df.apply(h_norm, col='h_scaled', axis=1)

    return df
