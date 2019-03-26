from __future__ import print_function
import os
import sys
import h5py
import mpld3    # create a scalable figure
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

num_timesteps = 1

col_names = ["duration","protocol_type","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

def load_dataset(dataset_dir, train_fname, test_fname):
    # attach the column names to the dataset

    # KDDTrain+_2.csv & KDDTest+_2.csv are the datafiles without the last column about the difficulty score
    # these have already been removed.
    df_train = pd.read_csv(os.path.join(dataset_dir, train_fname), header=None, names = col_names)
    df_test  = pd.read_csv(os.path.join(dataset_dir, test_fname), header=None, names = col_names)

    # shape, this gives the dimensions of the dataset
    print('Dimensions of the Training set:',df_train.shape)
    print('Dimensions of the Test set:', df_test.shape)

    # Label Distribution of Training and Test set
    # print('Label distribution of Training set:')
    # print(df_train['label'].value_counts())
    # print('-' * 50)
    # print('Label distribution of Test set:')
    # print(df_test['label'].value_counts())

    return df_train, df_test

def sort_dataframe(df_train, df_test):
    print(df_train['label'].value_counts())

# Encode non-categorical features
def encode_noncat_features(df_train, df_test):
    # 1. Identify categorical features

    # colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
    # explore categorical features
    print('Training set:')
    for col_name in df_train.columns:
        if df_train[col_name].dtypes == 'object' :
            unique_cat = len(df_train[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    # Test set
    print('Test set:')
    for col_name in df_test.columns:
        if df_test[col_name].dtypes == 'object' :
            unique_cat = len(df_test[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    # Conclusion: Need to make dummies for all categories as the distribution is fairly even. In total: 3+70+11=84 dummies.
    # Comparing the results shows that the Test set has fewer categories (6), these need to be added as empty columns.

    # 2. LabelEncoder : Insert categorical features into a 2D numpy array
    categorical_columns=['protocol_type', 'service', 'flag']
    # Get the categorical values into a 2D numpy array
    df_train_categorical_values = df_train[categorical_columns]
    df_test_categorical_values = df_test[categorical_columns]
    # print(df_train_categorical_values.head())

    # Make column names for dummies
    # protocol type
    unique_protocol = sorted(df_train.protocol_type.unique())
    unique_protocol = ["{prefix}_{proto}".format(prefix='Protocol_type', proto=proto) for proto in unique_protocol]
    # print("unique_protocol = %s" % unique_protocol)

    # service
    unique_service = sorted(df_train.service.unique())
    unique_service = ["{prefix}_{service}".format(prefix='service', service=service) for service in unique_service]
    # print("unique_service = %s" % unique_service)

    # flag
    unique_flag = sorted(df_train.flag.unique())
    unique_flag = ["{prefix}_{flag}".format(prefix='flag', flag=flag) for flag in unique_flag]

    # put together
    dummy_cols = unique_protocol + unique_service + unique_flag
    # print('dummy_cols = %s' % dummy_cols)

    # do same for test set
    unique_service_test = sorted(df_test.service.unique())
    unique_service_test = ["{prefix}_{service}".format(prefix='service', service=service) for service in unique_service_test]
    dummy_cols_test = unique_protocol + unique_service_test + unique_flag
    # print('dummy_cols_test = %s' % dummy_cols_test)

    # 3. Transform categorical features into numbers using LabelEncoder()
    df_train_categorical_values_enc = df_train_categorical_values.apply(LabelEncoder().fit_transform)
    # print(df_train_categorical_values_enc.head())
    # test set
    df_test_categorical_values_enc = df_test_categorical_values.apply(LabelEncoder().fit_transform)

    # 4. One-Hot-Encoding
    enc = OneHotEncoder()
    df_train_categorical_values_enc = enc.fit_transform(df_train_categorical_values_enc)
    df_train_cat_data = pd.DataFrame(df_train_categorical_values_enc.toarray(), columns=dummy_cols)
    # test set
    df_test_categorical_values_enc = enc.fit_transform(df_test_categorical_values_enc)
    df_test_cat_data = pd.DataFrame(df_test_categorical_values_enc.toarray(),columns=dummy_cols_test)

    # print(df_train_cat_data.head())

    # 5. Add 6 missing categories from train set to test set
    train_service = df_train['service'].tolist()
    test_service= df_test['service'].tolist()
    difference = list(set(train_service) - set(test_service))
    difference = ["{prefix}_{diff}".format(prefix='service', diff=diff) for diff in difference]
    # print(difference)
    for col in difference:
        df_test_cat_data[col] = 0

    # print(df_test_cat_data.shape)

    # 6. Join encoded categorical dataframe with the non-categorical dataframe
    df_train = df_train.join(df_train_cat_data)
    df_train.drop('flag', axis=1, inplace=True)
    df_train.drop('protocol_type', axis=1, inplace=True)
    df_train.drop('service', axis=1, inplace=True)

    # test data
    df_test = df_test.join(df_test_cat_data)
    df_test.drop('flag', axis=1, inplace=True)
    df_test.drop('protocol_type', axis=1, inplace=True)
    df_test.drop('service', axis=1, inplace=True)
    # print(df_test.head(5))
    print("New shape of training dataframe = ", df_train.shape)
    print("New shape of testing dataframe = ", df_test.shape)

    return df_train, df_test

def group_label_2cat(df_train, df_test, cate_type='Cat5'):
    df_train_label = df_train['label'].copy()
    df_test_label = df_test['label'].copy()
    # change the label column
    if(cate_type == 'Cat5'):
        df_train_label = df_train_label.replace({'normal' : 0, 'neptune' : 1 , 'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1, 'mailbomb': 1,
                                                'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1, 'ipsweep' : 2, 'nmap' : 2, 'portsweep' : 2,
                                                'satan' : 2, 'mscan' : 2, 'saint' : 2, 'ftp_write': 3, 'guess_passwd': 3, 'imap': 3,'multihop': 3,
                                                'phf': 3, 'spy': 3, 'warezclient': 3, 'warezmaster': 3, 'sendmail': 3, 'named': 3, 'snmpgetattack': 3,
                                                'snmpguess': 3, 'xlock': 3, 'xsnoop': 3, 'httptunnel': 3, 'buffer_overflow': 4, 'loadmodule': 4,
                                                'perl': 4, 'rootkit': 4, 'ps': 4, 'sqlattack': 4, 'xterm': 4})
        df_test_label = df_test_label.replace({'normal' : 0, 'neptune' : 1 , 'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
                                               'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1, 'ipsweep' : 2,
                                               'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2, 'ftp_write': 3, 'guess_passwd': 3,
                                               'imap': 3, 'multihop': 3, 'phf': 3, 'spy': 3, 'warezclient': 3, 'warezmaster': 3, 'sendmail': 3,
                                               'named': 3, 'snmpgetattack': 3, 'snmpguess': 3, 'xlock': 3, 'xsnoop': 3, 'httptunnel': 3,
                                               'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 'ps': 4, 'sqlattack': 4, 'xterm': 4})
        cate_group = 5
    else:
        label_dict = encode_label(df_train_label, df_test_label)
        if(cate_type == 'Cat2'):
            cate_group = 2
            for key in label_dict.keys():
                if key == 'normal': label_dict[key] = 0
                else: label_dict[key] = 1
            # print('labe_dict = ', label_dict)

        elif(cate_type == 'Cat40'):
            cate_group = 40
            label_dict = encode_label(df_train_label, df_test_label)
            print('labe_dict = ', label_dict)

        df_train_label = df_train_label.replace(label_dict)
        df_test_label = df_test_label.replace(label_dict)

    # put the new label column back
    df_train['label'] = df_train_label
    df_test['label'] = df_test_label
    # print(df_train['label'].head())

    print('Label distribution of Training set:')
    print(df_train['label'].value_counts())
    print('-' * 50)
    print('Label distribution of Test set:')
    print(df_test['label'].value_counts())
    return df_train, df_test, cate_group


def encode_label(dataframe_train, dataframe_test):
    label_set = set(dataframe_train.unique()) | set(dataframe_test.unique())
    # print(label_set, len(label_set))
    label_dict = {}
    id_label = 0
    for label in label_set:
        label_dict[label] = id_label
        id_label += 1
    return label_dict
    # print(list(dataframe.unique()))

def feature_scaling(features_arr, mode='train'):
    # Feature scaling : In order to avoid some features distances dominate others, we need to scale all of them.
    # Min-max normalisation is often known as feature scaling where the values of a numeric range of a feature of data, i.e. a property, are reduced to a scale between between 0 and 1.
    '''sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    '''
    print("{lines} {mode}: {print_info} {lines}".format(lines='+'*10, mode=mode, print_info="Feature scaling ... "))
    # features_arr = dataframe.as_matrix()
    print("features_arr shape = ", features_arr.shape)
    feature_size = features_arr.shape[2]
    # print(features_arr.reshape([-1, feature_size]).shape)
    features_arr = features_arr.reshape([-1, feature_size])
    print("features_arr after flatten : shape = ", features_arr.shape)
    features_arr_minmax = MinMaxScaler().fit_transform(features_arr)
    features_arr_minmax = features_arr_minmax.reshape([-1, num_timesteps, feature_size])
    print("features_arr after scaling : shape = ", features_arr_minmax.shape)
    return features_arr_minmax

def plot_dataset(fig_dir, dataframe, cate_group, stage='training'):
    dataframe_clone = dataframe.copy()

    dataframe_clone_label = dataframe_clone['label'].copy()
    dataframe_clone = dataframe_clone.drop('label', axis=1)

    if(cate_group == 5):    cate_dict = {0:'normal', 1:'DoS', 2:'Probe', 3:'R2L', 4:'U2R'}
    elif(cate_group == 2):  cate_dict = {0:'normal', 1:'attack'}

    target_classes_id = [key for key in sorted(cate_dict.keys())]
    target_classes = np.array([cate_dict[key] for key in sorted(cate_dict.keys())])
    X_data = dataframe_clone.as_matrix()
    Y_label = dataframe_clone_label.as_matrix()
    X_data = MinMaxScaler().fit_transform(X_data)
    # print(target_classes_id, target_classes)
    # label_dict = encode_label(dataframe_clone_label, df_test_clone_label)
    pca = PCA(n_components=2)
    # lda = LinearDiscriminantAnalysis(n_components=2)
    X_data_pca = pca.fit(X_data).transform(X_data)

    # Percentage of variance explained for each components
    print('Explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
    plt.figure()
    lw = 2
    for _id, target_class in zip(target_classes_id, target_classes):
        # print color, _id, target_class
        plt.scatter(X_data_pca[Y_label == _id, 0], X_data_pca[Y_label == _id, 1], alpha=.8, lw=lw,
                    label=target_class)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Category = %s : PCA visulization of NSL-KDD %s dataset after feature scaling' % (cate_group, stage))
    filename = "Category_{cate_group}-{prefix} dataset distribution after feature scaling-PCA Visulization.png".format(cate_group=str(cate_group), prefix=stage.title()) # str.title() : capitalize the first letter of the string
    plt.savefig(os.path.join(fig_dir, filename))
    # plt.show()

def split_by_timesteps(dataframe, mode):
    print('%s Split %s data now... %s' % ('='*15, mode, '='*15))
    labels = dataframe.label.unique()   # extract 'label' column
    print("length of labels : ", len(labels))
    categories = []
    label_list = []
    for label in labels:
        df_cate = dataframe.loc[dataframe['label'] == label]
        df_cate = df_cate.drop('label', axis=1)
        cate_np = df_cate.as_matrix()
        # print('shape 1 = ', cate_np.shape)
        reduced_length = (cate_np.shape[0] // num_timesteps) * num_timesteps
        cate_np = cate_np[:reduced_length, :]
        # print('shape 2 = ', cate_np.shape)
        group_np = cate_np.reshape([-1, num_timesteps, cate_np.shape[1]])
        print('shape group_np = ', group_np.shape)
        categories.extend(group_np)
        print('length categories = ', len(categories))
        label_list.extend([label for _ in range(group_np.shape[0])])
        print('length label_list = ', len(label_list))
        print('-'*50)
    data = np.array(categories)
    answer = np.array(label_list)
    print('shape data = %s\t shape answer = %s' % (data.shape, answer.shape))
    return data, answer

def df2csv(path, file_name, dataframe):
    if not os.path.exists(path):
        os.makedirs(path)
    dataframe.to_csv(os.path.join(path, file_name), index=False)

def main():
    dataset_dir = 'NSL-KDD_dataset'
    dataset_dir_processed = "NSL-KDD_dataset_processed"
    file_fmt = 'csv'
    fig_dir = 'NSL-KDD_dataset_figs'
    train_fname = "{fname}.{fmt}".format(fname="KDDTrain+_2", fmt=file_fmt)
    test_fname  = "{fname}.{fmt}".format(fname="KDDTest-21_2", fmt=file_fmt)
    df_train, df_test = load_dataset(dataset_dir, train_fname, test_fname)

    df_train, df_test = encode_noncat_features(df_train, df_test)

    # group labels to 5 categories
    df_train, df_test, cate_group = group_label_2cat(df_train, df_test, cate_type='Cat5')
    # return 0

    # encode labels to numbers(without categories)
    # label_dict = encode_label(df_train_label, df_test_label)

    # sort dataframe by label
    # print(df_train['label'].head(20))
    # print('-'*100)
    df_train.sort_values(["label"], inplace=True, ascending=True)
    df_test.sort_values(["label"], inplace=True, ascending=True)

    # plot dataset distribution here
    plot_dataset(fig_dir, df_train, cate_group, stage='training')
    plot_dataset(fig_dir, df_test, cate_group, stage='testing')
    print("Saving the dataset distribution figures successfully")

    X_train, Y_train = split_by_timesteps(df_train, 'Training')
    X_test,  Y_test  = split_by_timesteps(df_test,  'Testing')
    # print(X_train[:10])


    # # extract label column and data columns
    # df_train_label = df_train['label'].copy()
    # df_test_label = df_test['label'].copy()
    # # label_list = map(lambda dataframe: list(dataframe.unique()), df_train_label)
    #
    # # print(df_train['label'].head(20))
    # df_train.drop('label', axis=1, inplace=True)
    # df_test.drop('label', axis=1, inplace=True)
    # print("training set label types : \n", df_train_label.value_counts())
    # print("testing  set label types : \n", df_test_label.value_counts())

    # Feature scaling
    X_train = feature_scaling(X_train, mode="train")
    X_test = feature_scaling(X_test, mode="test")
    # print(X_train[:10])

    # label_dict = encode_label(df_train_label, df_test_label)
    # with open(os.path.join(dataset_dir_processed, 'label_dictionary.pkl'), 'wb') as fp:
    #     pickle.dump(label_dict, fp)
    # print("label_dict = %s,\t len(label_dict) = %s" % (label_dict, len(label_dict)))


    # Y_train = df_train_label.as_matrix()
    # Y_test = df_test_label.as_matrix()
    # Y_train = np.array(map(lambda item: label_dict[item], Y_train))
    # Y_test = np.array(map(lambda item: label_dict[item], Y_test))

    # one-hot encoding :
    # cate_group = 5
    Y_train_one_hot_enc = np.zeros((len(Y_train), cate_group))
    Y_train_one_hot_enc[np.arange(len(Y_train)), Y_train] = 1
    Y_test_one_hot_enc = np.zeros((len(Y_test), cate_group))
    Y_test_one_hot_enc[np.arange(len(Y_test)), Y_test] = 1
    # print(Y_test)

    # save huge numpy array to h5py
    hf = h5py.File(os.path.join(dataset_dir_processed, 'NSL-KDD_dataset_processed.h5'), 'w')
    hf.create_dataset('dataset_X_train', data=X_train)
    hf.create_dataset('dataset_X_test', data=X_test)
    hf.create_dataset('dataset_Y_train', data=Y_train_one_hot_enc)
    hf.create_dataset('dataset_Y_test', data=Y_test_one_hot_enc)
    hf.close()
    print("%s Saving to h5py successfully %s" % ('='*15, '='*15))


    # df_train_new_col_names = list(df_train.columns)
    # df_test_new_col_names  = list(df_test.columns)

    # saving to csv
    # train_fname = "{fname}_{suffix}.{fmt}".format(fname="KDDTrain+_2", suffix='processed', fmt=file_fmt)
    # test_fname = "{fname}_{suffix}.{fmt}".format(fname="KDDTest+_2", suffix='processed', fmt=file_fmt)
    # df2csv(dataset_dir_processed, train_fname, df_train)
    # df2csv(dataset_dir_processed, test_fname, df_test)

if __name__ == "__main__":
    main()
