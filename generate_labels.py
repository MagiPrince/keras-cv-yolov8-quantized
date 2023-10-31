import h5py
import pandas as pd
import numpy as np
import os

path_jet = "../CNNandMLP/data/jet/"

ETA = 2.4
PHI = 3.15
RADIUS = 0.4

GRANULARITY = 0.1
LEN_ETA = len(np.arange(start=-2.4, stop=2.4+GRANULARITY, step=GRANULARITY))
LEN_PHI = len(np.arange(start=-3.15, stop=3.15+GRANULARITY, step=GRANULARITY))
LEN_RADIUS = len(np.arange(start=0, stop=RADIUS+GRANULARITY, step=GRANULARITY))
SIZE_MATRIX = 64
WIDTH = int(round((LEN_RADIUS*2)/(LEN_ETA/SIZE_MATRIX)))
HEIGHT = int(round((LEN_RADIUS*2)/(LEN_PHI/SIZE_MATRIX)))


########################################################################
# Deal generated labels
########################################################################

labels_training = []
labels_validation = []
labels_test = []

classes_training = []
classes_validation = []
classes_test = []

########################################################################
# Jets
########################################################################

dir_files_jet = os.listdir(path_jet)
dir_files_jet = sorted(dir_files_jet)

for folder_element, jet_file in enumerate(dir_files_jet):
    print("--------------------- File : " + str(jet_file) + " ---------------------")

    f = h5py.File(os.path.join(path_jet, jet_file), "r")

    dset = f["caloCells"]

    # Creating a dictionnary of type parameter : index
    # data = dset["1d"]
    # array_indexes_to_convert = list(data.dtype.fields.keys())
    # dict_header_jet = {array_indexes_to_convert[i]: i for i in range(len(array_indexes_to_convert))}

    # Creating a dictionnary of type parameter : index
    data = dset["2d"]
    array_indexes_to_convert = list(data.dtype.fields.keys())
    dict_data_jet = {array_indexes_to_convert[i]: i for i in range(len(array_indexes_to_convert))}

    data_jet = dset["2d"][()]

    f.close()

    ########################################################################
    # Transform data
    ########################################################################

    for i in range(len(data_jet)):
        # print("--------------------- data_jet " + str(i) + " ---------------------")
        df_jet = pd.DataFrame(data_jet[i],  columns=list(dict_data_jet.keys()))

        df_jet['eta_rounded'] = df_jet['AntiKt4EMTopoJets_eta'].apply(lambda x : round(x*10)/10 if x == x else x)

        df_jet['phi_rounded'] = df_jet['AntiKt4EMTopoJets_phi'].apply(lambda x : round(x*10)/10 if x == x else x)

        tmp_labels = []
        tmp_classes = []
        # print(len(df_jet['eta_rounded']))

        for j in range(len(df_jet['eta_rounded'])):
            label = [59, 32, 10, 10]
            class_ = 0
            # Check if the value is not a NaN
            if df_jet['eta_rounded'].iloc[j] == df_jet['eta_rounded'].iloc[j]:
                # Check if the coordinate eta is in the covered range
                # print("ETA : "+str(df_jet['eta_rounded'].iloc[j]))
                if np.absolute(df_jet['eta_rounded'].iloc[j]) < 2 and np.absolute(df_jet['phi_rounded'].iloc[j]) < 2.75:

                    x = round((df_jet['eta_rounded'].iloc[j] + ETA) / (ETA*2) * (LEN_ETA-1)) #/ LEN_ETA
                    y = round((df_jet['phi_rounded'].iloc[j] + PHI) / (PHI*2) * (LEN_PHI-1)) #/ LEN_PHI

                    label = [x, y, 10, 10]
                    class_ = 1

            tmp_labels.append(label)
            tmp_classes.append(class_)

        # tmp_labels = sorted(tmp_labels, key=lambda x: x[0], reverse=False)

        if len(df_jet['eta_rounded']) < 30:
            for j in range(30-len(df_jet['eta_rounded'])):
                label = [55, 32, 10, 10]
                class_ = 0

                tmp_labels.append(label)
                tmp_classes.append(class_)

        if i%5 == 0:
            if i%10 == 0:
                labels_validation.append(tmp_labels)
                classes_validation.append(tmp_classes)
            else:
                labels_test.append(tmp_labels)
                classes_test.append(tmp_classes)
        else:
            labels_training.append(tmp_labels)
            classes_training.append(tmp_classes)

    # if folder_element+1 >= 4:
    #     break

np.save("labels_training.npy", np.array(labels_training))
np.save("labels_validation.npy", np.array(labels_validation))
np.save("labels_test.npy", np.array(labels_test))

np.save("classes_training.npy", np.array(classes_training))
np.save("classes_validation.npy", np.array(classes_validation))
np.save("classes_test.npy", np.array(classes_test))
