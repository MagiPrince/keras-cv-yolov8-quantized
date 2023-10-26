import h5py
import pandas as pd
import numpy as np
import os
import copy
import sys

path_jet = "../CNNandMLP/data/jet/"

NEURONS = 16

ETA = 2.4
PHI = 3.15
RADIUS = 0.4

GRANULARITY = 0.1
LEN_ETA = len(np.arange(start=-2.4, stop=2.4+GRANULARITY, step=GRANULARITY))
LEN_PHI = len(np.arange(start=-3.15, stop=3.15+GRANULARITY, step=GRANULARITY))
LEN_RADIUS = len(np.arange(start=0, stop=RADIUS+GRANULARITY, step=GRANULARITY))
SIZE_MATRIX = 64

SIZE_SIDE = np.sqrt(np.power(SIZE_MATRIX, 2)/NEURONS)
NB_NEURON_BY_SIDE = SIZE_MATRIX / SIZE_SIDE
MEDIAN = np.median(range(round(SIZE_SIDE)))

########################################################################
# Functions
########################################################################

def count_nb_available_neurons(matrix_of_closest_jet_coordinates):
    cnt = 0
    for i in range(round(NB_NEURON_BY_SIDE)):
        for j in range(round(NB_NEURON_BY_SIDE)):
            if (matrix_of_closest_jet_coordinates[i][j][-1] == 0):#.all():
                cnt += 1

    return cnt

def flatten_3d_array(matrix_of_closest_jet_coordinates):
    flattened_array = []
    for i in range(round(NB_NEURON_BY_SIDE)):
        for j in range(round(NB_NEURON_BY_SIDE)):
            flattened_array.append([round(matrix_of_closest_jet_coordinates[i][j][0]), round(matrix_of_closest_jet_coordinates[i][j][1]), 10, 10])#, matrix_of_closest_jet_coordinates[i][j][-1]])

    return flattened_array

def flatten_classes(matrix_of_closest_jet_class):
    flattened_array = []
    for i in range(round(NB_NEURON_BY_SIDE)):
        for j in range(round(NB_NEURON_BY_SIDE)):
            flattened_array.append(matrix_of_closest_jet_class[i][j])#, matrix_of_closest_jet_coordinates[i][j][-1]])

    return flattened_array

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

print("Median : " + str(MEDIAN))
print("Size side : " + str(SIZE_SIDE))
print("Size neuron side : " + str(NB_NEURON_BY_SIDE))

array_of_medians = np.arange(start=MEDIAN, stop=SIZE_MATRIX, step=SIZE_SIDE)

matrix_of_centers = np.zeros((round(NB_NEURON_BY_SIDE), round(NB_NEURON_BY_SIDE), 2))
for i in range(round(NB_NEURON_BY_SIDE)):
    for j in range(round(NB_NEURON_BY_SIDE)):
        matrix_of_centers[i][j] = [array_of_medians[j], array_of_medians[i]]

# print(matrix_of_centers)

for folder_element, jet_file in enumerate(dir_files_jet):
    print("--------------------- file_jet " + str(jet_file) + " ---------------------")

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
    for el in range(len(data_jet)):
        # print("--------------------- data_jet " + str(el) + " ---------------------")
        df_jet = pd.DataFrame(data_jet[el],  columns=list(dict_data_jet.keys()))

        df_jet['eta_rounded'] = df_jet['AntiKt4EMTopoJets_eta'].apply(lambda x : round(x*10)/10 if x == x else x)
        df_jet['phi_rounded'] = df_jet['AntiKt4EMTopoJets_phi'].apply(lambda x : round(x*10)/10 if x == x else x)

        tmp_labels = []
        tmp_classes = []
        cnt = 0
        # Create a matrix that will contain the position of the jet that each neuron should detect
        array = np.concatenate((matrix_of_centers, np.zeros((round(NB_NEURON_BY_SIDE), round(NB_NEURON_BY_SIDE), 1))), axis=-1)
        matrix_of_closest_jet_coordinates = np.ones((round(NB_NEURON_BY_SIDE), round(NB_NEURON_BY_SIDE), 4)) * [55, 32, 10, 10]
        matrix_of_closest_jet_class = np.zeros((round(NB_NEURON_BY_SIDE), round(NB_NEURON_BY_SIDE)))
        
        for j in range(len(df_jet['eta_rounded'])):
            # label = [55, 32, 0, 0, 0]
            # Check if the value is not a NaN
            if df_jet['eta_rounded'].iloc[j] == df_jet['eta_rounded'].iloc[j]:
                # Check if the coordinate eta is in the covered range
                if np.absolute(df_jet['eta_rounded'].iloc[j]) < 2 and np.absolute(df_jet['phi_rounded'].iloc[j]) < 2.75:

                    x = ((df_jet['eta_rounded'].iloc[j] + ETA) / (ETA*2) * (LEN_ETA-1))
                    y = ((df_jet['phi_rounded'].iloc[j] + PHI) / (PHI*2) * (LEN_PHI-1))
                    tmp_labels.append([x, y, 10, 10])
                    tmp_classes.append(1)

        # First step : Found closest jet to a neuron and if in range of the neuron assign it and remove jet from list
        for i in range(round(NB_NEURON_BY_SIDE)):
            for j in range(round(NB_NEURON_BY_SIDE)):
                dist_of_closest_jet = np.sqrt(pow(SIZE_SIDE/2-0, 2)+pow(SIZE_SIDE/2-0, 2))
                index_of_jet = -1
                for k in range(len(tmp_labels)):
                    # compute distance between center of neuron node and jet
                    dist = np.sqrt(pow(matrix_of_centers[i][j][0]-tmp_labels[k][0], 2)+pow(matrix_of_centers[i][j][1]-tmp_labels[k][1], 2))
                    if dist <= dist_of_closest_jet:
                        dist_of_closest_jet = dist
                        index_of_jet = k

                if index_of_jet != -1:
                    matrix_of_closest_jet_coordinates[i][j] = tmp_labels[index_of_jet]
                    matrix_of_closest_jet_class[i][j] = tmp_classes[index_of_jet]
                    del tmp_labels[index_of_jet]
                    del tmp_classes[index_of_jet]

        tmp_array = copy.deepcopy(matrix_of_closest_jet_coordinates)

        for k in range(len(tmp_labels)):
            if count_nb_available_neurons(matrix_of_closest_jet_coordinates) <= 0:
                break
            closest_neuron_index = [-1, -1]
            dist_of_closest_neuron = 64
            for i in range(round(NB_NEURON_BY_SIDE)):
                for j in range(round(NB_NEURON_BY_SIDE)):
                    if (matrix_of_closest_jet_class[i][j] == 0):# [55, 32]).all():
                        dist = np.sqrt(pow(matrix_of_centers[i][j][0]-tmp_labels[k][0], 2)+pow(matrix_of_centers[i][j][1]-tmp_labels[k][1], 2))
                        if dist <= dist_of_closest_neuron:
                            dist_of_closest_neuron = dist
                            closest_neuron_index = [i, j]

            matrix_of_closest_jet_coordinates[closest_neuron_index[0]][closest_neuron_index[1]] = tmp_labels[k]
            matrix_of_closest_jet_class[closest_neuron_index[0]][closest_neuron_index[1]] = 1

        # print(matrix_of_closest_jet_coordinates)
        # if not (tmp_array == matrix_of_closest_jet_coordinates).all():
        #     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix_of_centers]))
        #     print("------------------------")
        #     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in tmp_array]))
        #     print("------------------------")
        #     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix_of_closest_jet_coordinates]))
        #     print("-----")
        #     print(np.sqrt(pow(matrix_of_centers[0][0][0]-8.5, 2)+pow(matrix_of_centers[0][0][1]-0.5, 2)))
        #     sys.exit(1)

        # print(flatten_3d_array(matrix_of_closest_jet_coordinates))

        if el%5 == 0:
            if el%10 == 0:
                labels_validation.append(flatten_3d_array(matrix_of_closest_jet_coordinates))
                classes_validation.append(flatten_classes(matrix_of_closest_jet_class))
            else:
                labels_test.append(flatten_3d_array(matrix_of_closest_jet_coordinates))
                classes_test.append(flatten_classes(matrix_of_closest_jet_class))
        else:
            labels_training.append(flatten_3d_array(matrix_of_closest_jet_coordinates))
            classes_training.append(flatten_classes(matrix_of_closest_jet_class))

print(np.array(labels_training).shape)
print(np.array(classes_training).shape)
print(np.array(labels_validation).shape)
print(np.array(classes_validation).shape)

    # if folder_element+1 >= 4:
    #     break


np.save("labels_training_4_n_neurons.npy", np.array(labels_training))
np.save("labels_validation_4_n_neurons.npy", np.array(labels_validation))
np.save("labels_test_4_n_neurons.npy", np.array(labels_test))

np.save("classes_training_4_n_neurons.npy", np.array(classes_training))
np.save("classes_validation_4_n_neurons.npy", np.array(classes_validation))
np.save("classes_test_4_n_neurons.npy", np.array(classes_test))
