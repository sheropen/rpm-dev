# Adapted from Official evaluation script for SQuAD version 2.0.
# Link: https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py

# input: output file
# output: score file
# process: compare the output and answer in each of the test case

import collections
import re
import string
import json
from pathlib import Path

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
		return re.sub(regex, ' ', text)
	def white_space_fix(text):
		return ' '.join(text.split())
	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)
	def lower(text):
		return text.lower()
	return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
	if not s: return []
	return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
	return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
	gold_toks = get_tokens(a_gold)
	pred_toks = get_tokens(a_pred)
	common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
	num_same = sum(common.values())
	if len(gold_toks) == 0 or len(pred_toks) == 0:
		# If either is no-answer, then F1 is 1 if they agree, 0 otherwise
		return int(gold_toks == pred_toks)
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(pred_toks)
	recall = 1.0 * num_same / len(gold_toks)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

def main(args):

	# with open(args.input, 'r') as f:
	# 	data_list = json.load(f)
	# load .jsonl
	data_list = []
	with open(args.input, 'r') as f:
		for line in f:
			data_list.append(json.loads(line))
	output_list = []
	for data in data_list:
		# gold_answers = data['answer']
		qid = data[1]["qid"]
		category = data[1]["category"]
		gold_answers = data[1]["answer"]
		if not isinstance(gold_answers, list):
			gold_answers = [gold_answers]
		gold_answers = [str(a) for a in gold_answers]
		# output = data['output']
		output = data[0]["choices"][0]["message"]["content"]
		try:
			# extract the text between '[output_begin] ... [output_end]'
			# pattern = re.compile(r'\[output_begin\](.*)\[output_end\]')
			# search_results = pattern.search(output)
			# if search_results:
				# output = search_results.group(1)
			pred_answer = output
			# else:
			# 	pred_answer = output
		except AttributeError:
			pred_answer = ''
		exact_scores = max(compute_exact(a, pred_answer) for a in gold_answers)
		f1_scores = max(compute_f1(a, pred_answer) for a in gold_answers)
		# print(f'gold: {gold_answers}, pred: {pred_answer}, exact: {exact_scores}, f1: {f1_scores}')
		# data['prediction'] = pred_answer
		# data['scores'] = {
		# 	'exact': exact_scores,
		# 	'f1': f1_scores,
		# }
		output_list.append({
			"qid": qid,
			"category": category,
			"prompt": "\n\n\n\n",
			"answer": gold_answers,
			"output": output,
			"prediction": pred_answer,
			"scores": {
				"exact": exact_scores,
				"f1": f1_scores,
			}
		})
	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	with open(args.output, 'w') as f:
		json.dump(output_list, f, indent=2)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, required=True)
	parser.add_argument('--output', type=str, required=True)
	args = parser.parse_args()
	main(args)


# def get_raw_scores(dataset, preds):
# 	exact_scores = {}
# 	f1_scores = {}
# 	for article in dataset:
# 		for p in article['paragraphs']:
# 			for qa in p['qas']:
# 				qid = qa['id']
# 				gold_answers = [a['text'] for a in qa['answers']
# 												if normalize_answer(a['text'])]
# 				if not gold_answers:
# 					# For unanswerable questions, only correct answer is empty string
# 					gold_answers = ['']
# 				if qid not in preds:
# 					print('Missing prediction for %s' % qid)
# 					continue
# 				a_pred = preds[qid]
# 				# Take max over all gold answers
# 				exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
# 				f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
# 	return exact_scores, f1_scores

# def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
# 	new_scores = {}
# 	for qid, s in scores.items():
# 		pred_na = na_probs[qid] > na_prob_thresh
# 		if pred_na:
# 			new_scores[qid] = float(not qid_to_has_ans[qid])
# 		else:
# 			new_scores[qid] = s
# 	return new_scores

# def make_eval_dict(exact_scores, f1_scores, qid_list=None):
# 	if not qid_list:
# 		total = len(exact_scores)
# 		return collections.OrderedDict([
# 				('exact', 100.0 * sum(exact_scores.values()) / total),
# 				('f1', 100.0 * sum(f1_scores.values()) / total),
# 				('total', total),
# 		])
# 	else:
# 		total = len(qid_list)
# 		return collections.OrderedDict([
# 				('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
# 				('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
# 				('total', total),
# 		])

# def merge_eval(main_eval, new_eval, prefix):
# 	for k in new_eval:
# 		main_eval['%s_%s' % (prefix, k)] = new_eval[k]

# def make_qid_to_has_ans(dataset):
# 	qid_to_has_ans = {}
# 	for article in dataset:
# 		for p in article['paragraphs']:
# 			for qa in p['qas']:
# 				qid_to_has_ans[qa['id']] = bool(qa['answers'])
# 	return qid_to_has_ans
