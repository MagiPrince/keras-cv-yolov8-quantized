import os
import h5py
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

GRANULARITY = 0.1
SIZE_FINAL_MATRIX = 64

path_calocell = "../CNNandMLP/data/calocell/"

def normalization_method(val, method, min=0, max=0, mean=0, std=0):
    if method == 0:
        return val
    elif method == 1:
        return np.log(val)
    elif method == 2:
        return 2*((val-min)/(max-min))-1
    elif method == 3:
        return ((val-min)/(max-min))
    elif method == 4:
        if val > max:
            val = max
        return ((val-min)/(max-min))
    elif method == 5:
        return (val-mean)/std
    elif method == 6:
        return val/max
    elif method == 7:
        return 1-(mean+std)/((mean+std)+val)
    elif method == 8:
        return (np.power(2, val)-1)/(2-1)
    
    return 0


########################################################################
# Deal generated matrices
########################################################################

matrices_training = []
matrices_validation = []
matrices_test = []

########################################################################
# Transform data
########################################################################

dir_files = os.listdir(path_calocell)
dir_files = sorted(dir_files)

# Compute the len of phi and eta based on a given granularity
len_phi = len(np.arange(start=-3.15, stop=3.15+GRANULARITY, step=GRANULARITY))
len_eta = len(np.arange(start=-2.4, stop=2.4+GRANULARITY, step=GRANULARITY))

for folder_element, calocell_file in enumerate(dir_files):
    print("--------------------- File : " + str(calocell_file) + " ---------------------")

    f = h5py.File(os.path.join(path_calocell, calocell_file), "r")

    dset = f["caloCells"]

    data_calo = dset["2d"][('cell_E', 'cell_Sigma', 'cell_eta', 'cell_phi', 'cell_pt')]

    f.close()

    for i in range(0,len(data_calo)):
        # print("--------------------- data_calo " + str(i) + " ---------------------")
        df_calo = pd.DataFrame(data_calo[i],  columns=['cell_E', 'cell_Sigma', 'cell_eta', 'cell_phi', 'cell_pt'])

        # Check if the cell_E is above 2 sigma (that defined that the energy is interesting)
        # We keep the negative values
        df_calo['cell_EoverSigma'] = (df_calo["cell_E"]/df_calo['cell_Sigma'])
        df_calo = df_calo[df_calo['cell_EoverSigma'] > 2]
        df_calo[df_calo['cell_EoverSigma'] > 6] = 6

        df_calo['cell_eta_rounded'] = df_calo['cell_eta'].apply(lambda x : round(x*40)/40)

        df_calo['cell_phi_rounded'] = df_calo['cell_phi'].apply(lambda x : round(x*40)/40)
        

        matrix = np.zeros((3, SIZE_FINAL_MATRIX, SIZE_FINAL_MATRIX))

        for j in range(len(df_calo['cell_eta'])):
            if np.absolute(df_calo['cell_eta'].iloc[j]) <= 2.4:
                matrix[0][round((df_calo['cell_phi_rounded'].iloc[j] + 3.15) / (3.15*2) * (len_phi-1))][round((df_calo['cell_eta_rounded'].iloc[j] + 2.4) / (2.4*2) * (len_eta-1))] += df_calo["cell_EoverSigma"].iloc[j]
                matrix[1][round((df_calo['cell_phi_rounded'].iloc[j] + 3.15) / (3.15*2) * (len_phi-1))][round((df_calo['cell_eta_rounded'].iloc[j] + 2.4) / (2.4*2) * (len_eta-1))] += df_calo["cell_pt"].iloc[j]

                # y = (df_calo['cell_phi'].iloc[j] + 3.15) / (3.15*2) * (len_phi-1)
                # x = (df_calo['cell_eta'].iloc[j] + 2.4) / (2.4*2) * (len_eta-1)

                # if y < 64:
                #     coord_y = int(y)
                #     coord_x = int(x)

                #     float_y = y%1
                #     float_x = x%1

                #     matrix[0][coord_y][coord_x] += (df_calo['cell_EoverSigma'].iloc[j]) * (1-float_y) * (1-float_x)
                #     matrix[0][coord_y+1][coord_x] += (df_calo['cell_EoverSigma'].iloc[j]) * (float_y) * (1-float_x)
                #     matrix[0][coord_y][coord_x+1] += (df_calo['cell_EoverSigma'].iloc[j]) * (1-float_y) * (float_x)
                #     matrix[0][coord_y+1][coord_x+1] += (df_calo['cell_EoverSigma'].iloc[j]) * (float_y) * (float_x)

                #     matrix[1][coord_y][coord_x] += (df_calo['cell_pt'].iloc[j]) * (1-float_y) * (1-float_x)
                #     matrix[1][coord_y+1][coord_x] += (df_calo['cell_pt'].iloc[j]) * (float_y) * (1-float_x)
                #     matrix[1][coord_y][coord_x+1] += (df_calo['cell_pt'].iloc[j]) * (1-float_y) * (float_x)
                #     matrix[1][coord_y+1][coord_x+1] += (df_calo['cell_pt'].iloc[j]) * (float_y) * (float_x)
        

        val_min = matrix[0].min()
        val_max = matrix[0].max()
        val_mean = matrix[0].mean()
        val_std = matrix[0].std()

        val_min1 = matrix[1].min()
        val_max1 = matrix[1].max()
        val_mean1 = matrix[1].mean()
        val_std1 = matrix[1].std()

        for k in range(len(matrix[0])):
            for j in range(len(matrix[0][k])):
                matrix[0][k][j] = normalization_method(matrix[0][k][j], 7, val_min, val_max, val_mean, val_std)
                matrix[1][k][j] = normalization_method(matrix[1][k][j], 7, val_min1, val_max1, val_mean1, val_std1)

        matrix = np.ascontiguousarray(matrix.transpose(1,2,0))

        matrix *= 255
        matrix = matrix.astype(np.uint8)

        if i%5 == 0:
            if i%10 == 0:
                matrices_validation.append(matrix)
            else:
                matrices_test.append(matrix)
        else:
            matrices_training.append(matrix)

        # im = Image.fromarray(matrix, 'RGB')

        # plt.imshow(im)
        # plt.show()
    
    # if folder_element+1 >= 4:
    #     break

np.save("matrices_training.npy", np.array(matrices_training))
np.save("matrices_validation.npy", np.array(matrices_validation))
np.save("matrices_test.npy", np.array(matrices_test))