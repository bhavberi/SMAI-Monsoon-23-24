from test_util import *

if len(sys.argv) < 2:
    raise Exception("Data File Name not passed")

data_file = sys.argv[1]

try:
    fp = open(best_model_file_path_vit, "rb")
    best_model_vit = pickle.load(fp)
    fp.close()

    fp = open(best_model_file_path_resnet, "rb")
    best_model_resnet = pickle.load(fp)
    fp.close()
except Exception as e:
    print(e)
    raise Exception("Best Model loading error!!")

try:
    data = np.array(np.load(data_file, allow_pickle=True))
except Exception as e:
    print(e)
    raise Exception("Data loading error!!")

labels = data[:,3]

start_time = time.time()
pred_labels = best_model_vit.predict(data)
end_time = time.time()

scores = best_model_vit.scoring(labels, pred_labels)

print("VIT")
print("MODEL k: ", best_model_vit.k)
print("MODEL distance_metrics: ", best_model_vit.distance_metrics)
print("MODEL weights: ", best_model_vit.weights)
print("Accuracy: ", scores["accuracy"])
print("F1 Score: ", scores["f1"])
print("Precision: ", scores["precision"])
print("Recall: ", scores["recall"])
print("Time:", end_time - start_time, "s")

start_time = time.time()
pred_labels = best_model_resnet.predict(data)
end_time = time.time()

scores = best_model_resnet.scoring(labels, pred_labels)

print("RESNET")
print("MODEL k: ", best_model_resnet.k)
print("MODEL distance_metrics: ", best_model_resnet.distance_metrics)
print("MODEL weights: ", best_model_resnet.weights)
print("Accuracy: ", scores["accuracy"])
print("F1 Score: ", scores["f1"])
print("Precision: ", scores["precision"])
print("Recall: ", scores["recall"])
print("Time:", end_time - start_time, "s")