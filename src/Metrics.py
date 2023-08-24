import sys
import textdistance as td
import numpy as np

def evaluate(ground_truth, predictions):

	TP  = np.zeros((0,))
	FP  = np.zeros((0,))
	FN  = np.zeros((0,))
	LS_FP_FN = np.zeros((0,))
	GT_LENGTH = np.sum([1 if elem != '' else 0 for gt in ground_truth for elem in gt])
	PRED_LENGTH = np.sum([1 if elem != '' else 0 for pred in predictions for elem in pred])

	for gt, prediction in zip(ground_truth, predictions):

		gt_pred = [(a,b) for a in gt for b in prediction]
		levenshtein  = np.array([td.levenshtein.normalized_similarity(a, b) for (a,b) in gt_pred])
		identity  = np.array([td.identity.normalized_similarity(a, b) for (a,b) in gt_pred])
		print(gt_pred)
		
		n_pred = len(gt)
		idx = (-levenshtein).argsort()[:n_pred]
		for similarity, id_similarity in zip(levenshtein[idx], identity[idx]):
			if id_similarity == 1.0:
				TP = np.append(TP, 1)
			if id_similarity == 0:
				FN = np.append(FN, 1)	
				LS_FP_FN = np.append(LS_FP_FN, similarity)
	
	TP = np.sum(TP)
	FN = np.sum(FN)
	FP = PRED_LENGTH - TP
	print('TP:',TP)	
	print('FP:',FP)
	print('FN:',FN)		
	print('Accuracy:',TP/(TP+FP+FN))	
	print('Precision:',TP/(TP+FP))
	print('Recall:',TP/(TP+FN))		
	print('\n')
	print('LS_FP_FN:',np.mean(LS_FP_FN))
	return None    

if __name__ == '__main__':
	pass