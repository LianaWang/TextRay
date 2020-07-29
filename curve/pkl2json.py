import pickle
import mmcv
import json
import numpy as np
pkl_path = './cheby_maxr_center.pkl'
test_list = './data/ArT/ImageSets/Main/test.txt'

with open(pkl_path, 'rb') as f:
	pk = pickle.load(f)

img_ids = mmcv.list_from_file(test_list)

results = dict()
print(len(pk))
for i in range(len(pk)):
	js = "res_%d"%(int(img_ids[i])-5603)
	pkl = pk[i][np.where(pk[i][:, -1]>0.9)[0], :]
	result = []
	for j in range(len(pkl)):
		res = dict()
		res['points'] = pkl[j, :-1].reshape((-1, 2)).tolist()
		res['confidence'] = pkl[j, -1].tolist()
		result.append(res)
	results[js] = result

with open('/mnt/disk50/datasets/scene_text/cheby_maxr_center.json', 'w') as f1:
	json.dump(results, f1)
