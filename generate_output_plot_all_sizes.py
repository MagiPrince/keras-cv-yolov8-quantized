import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_average_precision_x_recall_curve_of_all_models_for_a_given_epoch_and_threshold(data, epoch, threshold, colors_model):
    models_array = []

    for i, model in enumerate(data.keys()):
        models_array.append(model)
        precision_array_all_iter = []
        recall_array_all_iter = []
        step = 0.025
        recall_thresholds = np.arange(start=0, stop=1+step, step=step)

        precision_a = []
        recall_a = []

        for iter in range(len(data[model])):

            prediction = data[model][str(iter)][str(epoch)]["prediction"][str(threshold)]

            nb_gt = data[model][str(iter)][str(epoch)]["values_computed"][-1]
            acc_TP = 0
            acc_FP = 0
            cnt_detection_reviewed = 0
            precision_array = []
            recall_array = []

            for element in prediction:
                cnt_detection_reviewed += 1
                if element[-1] == True:
                    acc_TP += 1
                else:
                    acc_FP += 1

                precision = acc_TP/cnt_detection_reviewed
                recall = acc_TP/nb_gt

                precision_array.append(precision)
                recall_array.append(recall)
            

            precision_a.append(precision_array)
            recall_a.append(recall_array)
            tmp_array_p = []
            tmp_array_r = []

            for t in recall_thresholds:
                if t <= max(recall_array):
                    tmp_array_p.append(precision_array[min(range(len(recall_array)), key=lambda ele: abs(recall_array[ele]-t))])
                else:
                    tmp_array_p.append(0)
                tmp_array_r.append(t)

            precision_array_all_iter.append(tmp_array_p)
            recall_array_all_iter.append(tmp_array_r)

        

        # print(precision_array_all_iter)

        precision_array_mean = np.array(precision_array_all_iter).mean(axis=0)
        recall_array_mean = np.array(recall_array_all_iter).mean(axis=0)

        # print(recall_array_mean)

        plt.plot(recall_array_mean, precision_array_mean, color=colors_model[i])
    # plt.plot(recall_a[0], precision_a[0])
    plt.legend(models_array)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Average Precision x Recall curve of all model sizes for epoch : " + str(epoch))
    plt.show()


def plot_average_precision_x_recall_curve(data, model, epoch, threshold):

    precision_array_all_iter = []
    recall_array_all_iter = []
    step = 0.025
    recall_thresholds = np.arange(start=0, stop=1+step, step=step)

    precision_a = []
    recall_a = []

    for iter in range(len(data[model])):

        prediction = data[model][str(iter)][str(epoch)]["prediction"][str(threshold)]

        nb_gt = data[model][str(iter)][str(epoch)]["values_computed"][-1]
        acc_TP = 0
        acc_FP = 0
        cnt_detection_reviewed = 0
        precision_array = []
        recall_array = []

        for element in prediction:
            cnt_detection_reviewed += 1
            if element[-1] == True:
                acc_TP += 1
            else:
                acc_FP += 1

            precision = acc_TP/cnt_detection_reviewed
            recall = acc_TP/nb_gt

            precision_array.append(precision)
            recall_array.append(recall)
        

        precision_a.append(precision_array)
        recall_a.append(recall_array)
        tmp_array_p = []
        tmp_array_r = []

        for t in recall_thresholds:
            if t <= max(recall_array):
                tmp_array_p.append(precision_array[min(range(len(recall_array)), key=lambda ele: abs(recall_array[ele]-t))])
            else:
                tmp_array_p.append(0)
            tmp_array_r.append(t)

        precision_array_all_iter.append(tmp_array_p)
        recall_array_all_iter.append(tmp_array_r)

    

    # print(precision_array_all_iter)

    precision_array_mean = np.array(precision_array_all_iter).mean(axis=0)
    recall_array_mean = np.array(recall_array_all_iter).mean(axis=0)

    # print(recall_array_mean)
    index = 4
    plt.plot(recall_array_mean, precision_array_mean)
    # plt.plot(recall_a[index], precision_a[index])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Average Precision x Recall curve")
    plt.show()


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

model_xs = np.load("YOLOv8_data_generated_backbone_xs_0_5_05_095.npy", allow_pickle=True).item()
model_xs.update(np.load("YOLOv8_data_generated_backbone_xs_5_10_05_095.npy", allow_pickle=True).item())

model_s = np.load("YOLOv8_data_generated_backbone_s_0_5_05_095.npy", allow_pickle=True).item()

model_m = np.load("YOLOv8_data_generated_backbone_m_0_5_05_095.npy", allow_pickle=True).item()

model_l = np.load("YOLOv8_data_generated_backbone_l_0_5_05_095.npy", allow_pickle=True).item()

model_xl = np.load("YOLOv8_data_generated_backbone_xl_0_5_05_095.npy", allow_pickle=True).item()

dict_of_models = {
    "model_xs": model_xs,
    "model_s": model_s,
    "model_m": model_m,
    "model_l": model_l,
    "model_xl": model_xl,
}

dict_results = {}

array_epochs = [2, 5, 10, 20, 30, 50, 100, 300, 500]
colors_model = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for model in dict_of_models.keys():

    dict_results[model] = {}

    lossess_array = []
    val_lossess_array = []

    f1_scores_array = []
    true_positifs_array = []
    false_positifs_array = []
    false_negatives_array = []

    ap_50_array = []
    ap_50_95_array = []

    for iteration in dict_of_models[model]:
        f1_scores_of_epochs = []
        true_positifs_of_epochs = []
        false_positifs_of_epochs = []
        false_negatives_of_epochs = []
        ap_50_of_epochs = []
        ap_50_95_of_epochs = []
        for epoch in dict_of_models[model][iteration]:
            if epoch not in ['loss', 'val_loss']:
                f1_scores_of_epochs.append(dict_of_models[model][iteration][epoch]['values_computed'][3])
                true_positifs_of_epochs.append(dict_of_models[model][iteration][epoch]['values_computed'][0])
                false_positifs_of_epochs.append(dict_of_models[model][iteration][epoch]['values_computed'][1])
                false_negatives_of_epochs.append(dict_of_models[model][iteration][epoch]['values_computed'][2])
                ap_50_of_epochs.append(compute_average_precision_50_95(dict_of_models[model], iteration, epoch, True))
                ap_50_95_of_epochs.append(compute_average_precision_50_95(dict_of_models[model], iteration, epoch))
            else:
                lossess_array.append(dict_of_models[model][iteration]['loss'])
                val_lossess_array.append(dict_of_models[model][iteration]['val_loss'])

        f1_scores_array.append(f1_scores_of_epochs)
        true_positifs_array.append(true_positifs_of_epochs)
        false_positifs_array.append(false_positifs_of_epochs)
        false_negatives_array.append(false_negatives_of_epochs)
        ap_50_array.append(ap_50_of_epochs)
        ap_50_95_array.append(ap_50_95_of_epochs)

    dict_results[model]["f1_scores_array"] = f1_scores_array
    dict_results[model]["true_positifs_array"] = true_positifs_array
    dict_results[model]["false_positifs_array"] = false_positifs_array
    dict_results[model]["false_negatives_array"] = false_negatives_array
    dict_results[model]["ap_50_array"] = ap_50_array
    dict_results[model]["ap_50_95_array"] = ap_50_95_array
    

plot_average_precision_x_recall_curve(dict_of_models, "model_xs", 100, 0.5)
# plot_average_precision_x_recall_curve_of_all_models_for_a_given_epoch_and_threshold(dict_of_models, 100, 0.5, colors_model)


########################################################################################################################
# Pandas output of the data
########################################################################################################################

stats_dict = {}

for model in dict_of_models.keys():

    print("------------------------------------- Displaying model : " + model + " -------------------------------------")

    dict_results[model]["precision_computed"] = [[dict_results[model]["true_positifs_array"][i][j]/(dict_results[model]["true_positifs_array"][i][j] + dict_results[model]["false_positifs_array"][i][j]) for j in range(len(dict_results[model]["true_positifs_array"][i]))]
                        for i in range(len(dict_results[model]["true_positifs_array"]))]

    dict_results[model]["recall_computed"] = [[dict_results[model]["true_positifs_array"][i][j]/(dict_results[model]["true_positifs_array"][i][j] + dict_results[model]["false_negatives_array"][i][j]) for j in range(len(dict_results[model]["false_negatives_array"][i]))]
                        for i in range(len(dict_results[model]["true_positifs_array"]))]

    stats_dict[model] = {
        "epoch": np.array(array_epochs),
        "F1 score max": np.array(dict_results[model]["f1_scores_array"]).max(axis=0),
        "F1 score min": np.array(dict_results[model]["f1_scores_array"]).min(axis=0),
        "F1 score mean": np.array(dict_results[model]["f1_scores_array"]).mean(axis=0),
        "F1 score std": np.array(dict_results[model]["f1_scores_array"]).std(axis=0),
        "Precision max": np.array(dict_results[model]["precision_computed"]).max(axis=0),
        "Precision min": np.array(dict_results[model]["precision_computed"]).min(axis=0),
        "Precision mean": np.array(dict_results[model]["precision_computed"]).mean(axis=0),
        "Precision std": np.array(dict_results[model]["precision_computed"]).std(axis=0),
        "Recall max": np.array(dict_results[model]["recall_computed"]).max(axis=0),
        "Recall min": np.array(dict_results[model]["recall_computed"]).min(axis=0),
        "Recall mean": np.array(dict_results[model]["recall_computed"]).mean(axis=0),
        "Recall std": np.array(dict_results[model]["recall_computed"]).std(axis=0),
        "AP[.5] max": np.array(dict_results[model]["ap_50_array"]).max(axis=0),
        "AP[.5] min": np.array(dict_results[model]["ap_50_array"]).min(axis=0),
        "AP[.5] mean": np.array(dict_results[model]["ap_50_array"]).mean(axis=0),
        "AP[.5] std": np.array(dict_results[model]["ap_50_array"]).std(axis=0),
        "AP[.5,0.05,0.95] max": np.array(dict_results[model]["ap_50_95_array"]).max(axis=0),
        "AP[.5,0.05,0.95] min": np.array(dict_results[model]["ap_50_95_array"]).min(axis=0),
        "AP[.5,0.05,0.95] mean": np.array(dict_results[model]["ap_50_95_array"]).mean(axis=0),
        "AP[.5,0.05,0.95] std": np.array(dict_results[model]["ap_50_95_array"]).std(axis=0),
        }

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    df = pd.DataFrame(data=stats_dict[model])
    print(df)

########################################################################################################################
# Computing and plotting Precision and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("Precision")
plt.title("Precision over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["Precision mean"]-stats_dict[model]["Precision std"], stats_dict[model]["Precision mean"]+stats_dict[model]["Precision std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["Precision mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='center right')

plt.show()

########################################################################################################################
# Computing and plotting Recall and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("Recall")
plt.title("Recall over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["Recall mean"]-stats_dict[model]["Recall std"], stats_dict[model]["Recall mean"]+stats_dict[model]["Recall std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["Recall mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='center right')

plt.show()

########################################################################################################################
# Computing and plotting F1 score and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("F1 score")
plt.title("F1 score over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["F1 score mean"]-stats_dict[model]["F1 score std"], stats_dict[model]["F1 score mean"]+stats_dict[model]["F1 score std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["F1 score mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='center right')

plt.show()

########################################################################################################################
# Computing and plotting AP[.5] and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("AP[.5]")
plt.title("AP[.5] over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["AP[.5] mean"]-stats_dict[model]["AP[.5] std"], stats_dict[model]["AP[.5] mean"]+stats_dict[model]["AP[.5] std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["AP[.5] mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='center right')

plt.show()

########################################################################################################################
# Computing and plotting AP[.5] and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("AP[.5,0.05,0.95]")
plt.title("AP[.5,0.05,0.95] over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["AP[.5,0.05,0.95] mean"]-stats_dict[model]["AP[.5,0.05,0.95] std"], stats_dict[model]["AP[.5,0.05,0.95] mean"]+stats_dict[model]["AP[.5,0.05,0.95] std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["AP[.5,0.05,0.95] mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='center right')

plt.show()