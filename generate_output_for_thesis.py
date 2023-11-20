import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_average_precision_50_95(data_generated, iteration, epoch, compute_only_50=False):
    prediction = data_generated[iteration][epoch]["prediction"]

    ap = []

    for key in prediction:

        nb_gt = data_generated[iteration][epoch]["values_computed"][-1]
        acc_TP = 0
        acc_FP = 0
        cnt_detection_reviewed = 0
        precision_array = []
        recall_array = []

        for element in prediction[key]:
            cnt_detection_reviewed += 1
            if element[-1] == True:
                acc_TP += 1
            else:
                acc_FP += 1

            precision = acc_TP/cnt_detection_reviewed
            recall = acc_TP/nb_gt

            precision_array.append(precision)
            recall_array.append(recall)

        ap.append(np.trapz(precision_array, recall_array))
        if compute_only_50:
            break

    return np.mean(ap)


# data_generated = np.load("YOLOv8_data_generated_backbone_xs_0_5_05_095.npy", allow_pickle=True).item()
# data_generated.update(np.load("YOLOv8_data_generated_backbone_xs_5_10_05_095.npy", allow_pickle=True).item())

data_generated = np.load("YOLOv8_data_generated_backbone_xs_reduced_16x_0_5_05_095.npy", allow_pickle=True).item()

array_epochs = [2, 5, 10, 20, 30, 50, 100, 300, 500]

lossess_array = []
val_lossess_array = []

f1_scores_array = []
true_positifs_array = []
false_positifs_array = []
false_negatives_array = []

ap_50_array = []
ap_50_95_array = []

for iteration in data_generated:
    f1_scores_of_epochs = []
    true_positifs_of_epochs = []
    false_positifs_of_epochs = []
    false_negatives_of_epochs = []
    ap_50_of_epochs = []
    ap_50_95_of_epochs = []
    for epoch in data_generated[iteration]:
        if epoch not in ['loss', 'val_loss']:
            f1_scores_of_epochs.append(data_generated[iteration][epoch]['values_computed'][3])
            true_positifs_of_epochs.append(data_generated[iteration][epoch]['values_computed'][0])
            false_positifs_of_epochs.append(data_generated[iteration][epoch]['values_computed'][1])
            false_negatives_of_epochs.append(data_generated[iteration][epoch]['values_computed'][2])
            ap_50_of_epochs.append(compute_average_precision_50_95(data_generated, iteration, epoch, True))
            ap_50_95_of_epochs.append(compute_average_precision_50_95(data_generated, iteration, epoch))
        else:
            lossess_array.append(data_generated[iteration]['loss'])
            val_lossess_array.append(data_generated[iteration]['val_loss'])

    f1_scores_array.append(f1_scores_of_epochs)
    true_positifs_array.append(true_positifs_of_epochs)
    false_positifs_array.append(false_positifs_of_epochs)
    false_negatives_array.append(false_negatives_of_epochs)
    ap_50_array.append(ap_50_of_epochs)
    ap_50_95_array.append(ap_50_95_of_epochs)

# print("F1 score max : " + str(np.array(f1_scores_array).max(axis=0)) + "\n / min : " + str(np.array(f1_scores_array).min(axis=0))
#       + "\n / mean : " + str(np.array(f1_scores_array).mean(axis=0)) + "\n / std : " + str(np.array(f1_scores_array).std(axis=0)))

precision_computed = [[true_positifs_array[i][j]/max(((true_positifs_array[i][j] + false_positifs_array[i][j]), 1)) for j in range(len(true_positifs_array[i]))]
                      for i in range(len(true_positifs_array))]

recall_computed = [[true_positifs_array[i][j]/max(((true_positifs_array[i][j] + false_negatives_array[i][j]), 1)) for j in range(len(false_negatives_array[i]))]
                      for i in range(len(true_positifs_array))]

stats_dict = {
    "epoch": np.array(array_epochs),
    "F1 score max": np.array(f1_scores_array).max(axis=0),
    "F1 score min": np.array(f1_scores_array).min(axis=0),
    "F1 score mean": np.array(f1_scores_array).mean(axis=0),
    "F1 score std": np.array(f1_scores_array).std(axis=0),
    "Precision max": np.array(precision_computed).max(axis=0),
    "Precision min": np.array(precision_computed).min(axis=0),
    "Precision mean": np.array(precision_computed).mean(axis=0),
    "Precision std": np.array(precision_computed).std(axis=0),
    "Recall max": np.array(recall_computed).max(axis=0),
    "Recall min": np.array(recall_computed).min(axis=0),
    "Recall mean": np.array(recall_computed).mean(axis=0),
    "Recall std": np.array(recall_computed).std(axis=0),
    "AP[.5] max": np.array(ap_50_array).max(axis=0),
    "AP[.5] min": np.array(ap_50_array).min(axis=0),
    "AP[.5] mean": np.array(ap_50_array).mean(axis=0),
    "AP[.5] std": np.array(ap_50_array).std(axis=0),
    "AP[.5,0.05,0.95] max": np.array(ap_50_95_array).max(axis=0),
    "AP[.5,0.05,0.95] min": np.array(ap_50_95_array).min(axis=0),
    "AP[.5,0.05,0.95] mean": np.array(ap_50_95_array).mean(axis=0),
    "AP[.5,0.05,0.95] std": np.array(ap_50_95_array).std(axis=0),
    }

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.DataFrame(data=stats_dict)
print(df)

########################################################################################################################
# Computing and plotting F1 scores over epochs
########################################################################################################################

f1_scores_mean = np.array(f1_scores_array).mean(axis=0)
f1_scores_std = np.array(f1_scores_array).std(axis=0)

# plt.plot(array_epochs, f1_scores_mean, marker="o", markersize=3)
# plt.fill_between(array_epochs, f1_scores_mean-f1_scores_std, f1_scores_mean+f1_scores_std, alpha=0.3, color='C0')
# # plt.figure()
# # plt.errorbar(array_epochs, f1_scores_mean, yerr=f1_scores_std, marker='o', markersize=3, capsize=3)

# plt.xlabel("Epochs")
# plt.ylabel("F1 score")
# plt.title("F1 score mean and std over the epochs")

# plt.show()

########################################################################################################################
# Computing and plotting losses mean and std for training and validation over epochs
########################################################################################################################

losses_mean = np.array(lossess_array).mean(axis=0)
losses_std = np.array(lossess_array).std(axis=0)

val_losses_mean = np.array(val_lossess_array).mean(axis=0)
val_losses_std = np.array(val_lossess_array).std(axis=0)

# # Generating plot

# f, a = plt.subplots()

# a.fill_between(range(len(losses_mean)), losses_mean-losses_std, losses_mean+losses_std, alpha=0.3, color='C0')
# p1 = a.plot(range(len(losses_mean)), losses_mean, color='C0')
# p2 = a.fill(np.NaN, np.NaN, 'C0', alpha=0.3)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Loss mean and std over the epochs")

# a.fill_between(range(len(val_losses_mean)), val_losses_mean-val_losses_std, val_losses_mean+val_losses_std, alpha=0.3, color='#ff7f0e')
# p3 = a.plot(range(len(val_losses_mean)), val_losses_mean, color='#ff7f0e')
# p4 = a.fill(np.NaN, np.NaN, '#ff7f0e', alpha=0.3)
# a.legend([(p2[0], p1[0]), (p3[0], p4[0]), ], ['Training loss', "Validation loss"])

# plt.show()

########################################################################################################################
# Computing and plotting f1 score and losses mean and std for training and validation over epochs
########################################################################################################################

f, a = plt.subplots()

# Training loss

a.fill_between(range(len(losses_mean)), losses_mean-losses_std, losses_mean+losses_std, alpha=0.3, color='C0')
p1 = a.plot(range(len(losses_mean)), losses_mean, color='C0')
p2 = a.fill(np.NaN, np.NaN, 'C0', alpha=0.3)
a.set_xlabel("Epochs")
a.set_ylabel("Loss")
plt.title("F1 score / Loss mean and std over the epochs")

# Validation loss

a.fill_between(range(len(val_losses_mean)), val_losses_mean-val_losses_std, val_losses_mean+val_losses_std, alpha=0.3, color='#ff7f0e')
p3 = a.plot(range(len(val_losses_mean)), val_losses_mean, color='#ff7f0e')
p4 = a.fill(np.NaN, np.NaN, '#ff7f0e', alpha=0.3)

# F1 score

b = a.twinx()

b.fill_between(array_epochs, f1_scores_mean-f1_scores_std, f1_scores_mean+f1_scores_std, alpha=0.3, color='#2ca02c')
p5 = b.plot(array_epochs, f1_scores_mean, marker="o", markersize=3, color='#2ca02c')
p6 = b.fill(np.NaN, np.NaN, '#2ca02c', alpha=0.3)
b.legend([(p5[0], p6[0]), (p2[0], p1[0]), (p3[0], p4[0]), ], ['F1 score', 'Training loss', "Validation loss"], loc='center right')
b.set_ylabel("F1 score", color='#2ca02c')

plt.show()