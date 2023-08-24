from test_util import *

if len(sys.argv) < 2:
    raise Exception("Data File Name not passed")

data_file = sys.argv[1]

try:
    fp = open(best_model_file_path, "rb")
    best_model = pickle.load(fp)
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
pred_labels = best_model.predict(data)
end_time = time.time()

scores = best_model.scoring(labels, pred_labels)

print("Accuracy: ", scores["accuracy"])
print("F1 Score: ", scores["f1"])
print("Precision: ", scores["precision"])
print("Recall: ", scores["recall"])
print("Time:", end_time - start_time, "s")