import numpy as np
import matplotlib.pyplot as plt

data = np.load("YOLOv8_data_generated_backbone_xs_0_5_05_095.npy", allow_pickle=True).item()

prediction = data["0"]["100"]["prediction"]["0.95"]

for key in prediction:
    print(key)

nb_gt = data["0"]["100"]["values_computed"][-1]
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

print(np.trapz(precision_array, recall_array))

p = []
r = []

prec_at_rec = []
for recall_level in np.linspace(0.0, 1.0, 11):
    try:
        args = np.argwhere(recall_array >= recall_level).flatten()
        r.append(recall_level)
        prec = precision_array[min(args)]
        p.append(prec)
    except ValueError:
        prec = 0.0
        p.append(prec)
    prec_at_rec.append(prec)
avg_prec = np.mean(prec_at_rec)

print(r)
print(p)

print(avg_prec)

plt.plot(recall_array, precision_array)
plt.plot(r, p, "--")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision x Recall curve")
plt.show()

