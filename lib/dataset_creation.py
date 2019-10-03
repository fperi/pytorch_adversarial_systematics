import pandas as pd
import numpy as np
import math


def get_normal(mean, std, angle, n_samples):

    '''
    Function creates a sample of points distributed
    as a 2d normal distribution

    :param mean: mean of the desired 2d normal
    :param std: std of the desired 2d normal
    :param angle: optional rotation angle
    :param n_samples: how many samples to generate

    :return: numpy array of the x,y coordinates of the 2d normal
    '''

    rMatrix = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    output = np.array(mean) + np.array(std) * np.random.randn(n_samples, 2)
    output = output.dot(rMatrix)

    return output


def get_annulus(radius, std, n_samples):

    '''
    Function creates a sample of points distributed
    as an annulus

    :param radius: radius of the desired annulus
    :param std: std of the desired annulus
    :param n_samples: how many samples to generate

    :return: numpy array of the x,y coordinates of the annulus
    '''

    output = []

    for i in range(n_samples):
        rad = (radius + np.random.normal(radius, std))
        angle = np.random.uniform(0, 2*math.pi)
        x, y = rad * math.cos(angle), rad * math.sin(angle)
        output.append([x, y])

    return np.array(output)


def create_dataset(n_samples, features):

    '''
    The dataset is a pandas dataframe with the following columns:
        - x: x coordinate of the distribution
        - y: y coordinate of the distribution
        - class_idx: the index associated to the classes
        - class_name: class name (gaus, gaus_s1, gaus_s2, annulus)
        - class_color: color to be used in the plots for this class
        - syst_idx: the index associated to the systematic variations
        - test_flag: flags for the subset to be used as test dataset

    :return: dataframe
    '''

    # set seeds
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    # define column names
    column_names = features[:]
    column_names.extend(['class_idx'])

    # create default 2d normal mean=[0,0], std=[1,2]
    df_gaus = pd.DataFrame(np.zeros((n_samples, 3)),
                           columns=[column_names])
    df_gaus[features] = get_normal([0, 0], [1, 2], 0, n_samples)
    df_gaus['class_name'] = 'gaus'
    df_gaus['class_color'] = 'salmon'
    df_gaus['syst_idx'] = 1
    df_gaus['test_flag'] = np.random.randint(10, size=len(df_gaus)) == 0

    # create systematic 2d normal mean=[0,-2], std=[1,2]
    df_gaus_s1 = pd.DataFrame(np.zeros((n_samples, 3)),
                              columns=[column_names])
    df_gaus_s1[features] = get_normal([0, -2], [1, 2], 0, n_samples)
    df_gaus_s1['class_name'] = 'gaus_s1'
    df_gaus_s1['class_color'] = 'lightskyblue'
    df_gaus_s1['syst_idx'] = 2
    df_gaus_s1['test_flag'] = np.random.randint(10, size=len(df_gaus_s1)) == 0

    # create systematic 2d normal mean=[0,2], std=[1,2], rot=0
    df_gaus_s2 = pd.DataFrame(np.zeros((n_samples, 3)),
                              columns=[column_names])
    df_gaus_s2[features] = get_normal([0, 2], [1, 2], 0, n_samples)
    df_gaus_s2['class_name'] = 'gaus_s2'
    df_gaus_s2['class_color'] = 'limegreen'
    df_gaus_s2['syst_idx'] = 3
    df_gaus_s2['test_flag'] = np.random.randint(10, size=len(df_gaus_s2)) == 0

    # create annulus radius 2 and std 1
    df_an = pd.DataFrame(np.ones((n_samples, 3)),
                         columns=[column_names])
    df_an[features] = get_annulus(2, 1, n_samples)
    df_an['class_name'] = 'annulus'
    df_an['class_color'] = 'grey'
    df_an['syst_idx'] = 0
    df_an['test_flag'] = np.random.randint(10, size=len(df_an)) == 0

    # merge dataframes
    df_tot = pd.concat([df_gaus, df_an, df_gaus_s1, df_gaus_s2])
    df_tot = df_tot.sample(frac=1, random_state=RANDOM_STATE)
    df_tot.index = range(len(df_tot))
    df_tot.columns = df_tot.columns.get_level_values(0)

    return df_tot
