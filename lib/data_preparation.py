import numpy as np
from imblearn.over_sampling import RandomOverSampler
from random import shuffle
import torch


def over_sample(X, Y, RANDOM_STATE):

    '''
    Function rebalances classes based on Y values

    :param X: array of features
    :param Y: labels of classes
    :param RANDOM_STATE: a seed for the RANDOM STATE

    :return: rebalanced X and Y
    '''

    # initialise oversampler
    ros = RandomOverSampler(random_state=RANDOM_STATE)

    # rebalance classes and shuffle
    X, Y = ros.fit_resample(X, Y)
    XY = list(zip(X, Y))
    shuffle(XY)
    X, Y = zip(*XY)

    return X, Y


def prepare_data(df, features, network_to_train, classes_to_exclude=[]):

    '''
    Function prepares dataset for training.

    :param df: input dataframe
    :param features: list of the training features
    :param network_to_train: for which neural network to prepare the dataset
        for the discriminant we need all classes plus the systematic
        uncertainties, we also need to rebalance the classes;
        for the adversarial term we do not need the annulus and
        we don't need to rebalance anything
    :param classes_to_exclude: which classes to exclude from the dataset

    :return: modified dataframe + training/testing features and label
    '''

    # set seeds
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # initialise columns where predictions will be stored
    df['predictionD'] = np.random.uniform(0, 1, len(df))
    df['predictionA'] = np.random.rand(len(df), 4).tolist()

    # transform input in tensors
    X_train = \
        (df.loc[(~df['test_flag']) &
                (~df['class_name'].isin(classes_to_exclude)),
                features]).values
    Y_train = \
        (df.loc[(~df['test_flag']) &
                (~df['class_name'].isin(classes_to_exclude)),
                'class_idx']).values
    X_test = \
        (df.loc[(df['test_flag']) &
                (~df['class_name'].isin(classes_to_exclude)),
                features]).values
    Y_test = \
        (df.loc[(df['test_flag']) &
                (~df['class_name'].isin(classes_to_exclude)),
                'class_idx']).values

    AY_train = \
        (df.loc[(~df['test_flag']) &
                (~df['class_name'].isin(classes_to_exclude)),
                'syst_idx']).values
    AY_test = \
        (df.loc[(df['test_flag']) &
                (~df['class_name'].isin(classes_to_exclude)),
                'syst_idx']).values

    # need to oversample minority class for the discriminant
    if network_to_train == 'D':
        XA = np.column_stack((X_train, AY_train))
        XA, Y_train = over_sample(XA, Y_train, RANDOM_STATE)
        X_train = np.array(XA)[:, :2].astype(float)
        AY_train = np.array(XA)[:, 2].reshape(-1, 1)

        XA = np.column_stack((X_test, AY_test))
        XA, Y_test = over_sample(XA, Y_test, RANDOM_STATE)
        X_test = np.array(XA)[:, :2].astype(float)
        AY_test = np.array(XA)[:, 2].reshape(-1, 1)

    DX_train_t = torch.Tensor(np.array(X_train))
    DY_train_t = torch.Tensor(np.array(Y_train)).reshape(-1, 1)
    DX_test_t = torch.Tensor(np.array(X_test))
    DY_test_t = torch.Tensor(np.array(Y_test)).reshape(-1, 1)
    DAY_train_t = torch.Tensor(np.array(AY_train)).reshape(-1, 1)
    DAY_test_t = torch.Tensor(np.array(AY_test)).reshape(-1, 1)

    dataset = [DX_train_t, DY_train_t,
               DX_test_t, DY_test_t,
               DAY_train_t, DAY_test_t]

    return df, dataset
