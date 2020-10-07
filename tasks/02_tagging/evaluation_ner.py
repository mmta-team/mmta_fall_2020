from collections import OrderedDict

def _update_chunk(candidate, prev, current_tag, current_chunk, current_pos, prediction=False):
    if candidate == 'B-' + current_tag:
        if len(current_chunk) > 0 and len(current_chunk[-1]) == 1:
                current_chunk[-1].append(current_pos - 1)
        current_chunk.append([current_pos])
    elif candidate == 'I-' + current_tag:
        if prediction and (current_pos == 0 or current_pos > 0 and prev.split('-', 1)[-1] != current_tag):
            current_chunk.append([current_pos])
        if not prediction and (current_pos == 0 or current_pos > 0 and prev == 'O'):
            current_chunk.append([current_pos])
    elif current_pos > 0 and prev.split('-', 1)[-1] == current_tag:
        if len(current_chunk) > 0:
            current_chunk[-1].append(current_pos - 1)

def _update_last_chunk(current_chunk, current_pos):
    if len(current_chunk) > 0 and len(current_chunk[-1]) == 1:
        current_chunk[-1].append(current_pos - 1)

def _tag_precision_recall_f1(tp, fp, fn):
    precision, recall, f1 = 0, 0, 0
    if tp + fp > 0:
        precision = tp / (tp + fp) * 100
    if tp + fn > 0:
        recall = tp / (tp + fn) * 100
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def _aggregate_metrics(results, total_correct):
    total_true_entities = 0
    total_predicted_entities = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for tag, tag_metrics in results.items():
        n_pred = tag_metrics['n_predicted_entities']
        n_true = tag_metrics['n_true_entities']
        total_true_entities += n_true
        total_predicted_entities += n_pred
        total_precision += tag_metrics['precision'] * n_pred
        total_recall += tag_metrics['recall'] * n_true
    accuracy = total_correct / total_true_entities * 100
    total_precision = total_precision / total_predicted_entities if total_predicted_entities != 0 else 0
    total_recall = total_recall / total_true_entities
    if total_precision + total_recall > 0:
        if total_precision + total_recall >= 1e-16:
            total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
        else:
            total_f1 = 0
    return total_true_entities, total_predicted_entities, \
           total_precision, total_recall, total_f1, accuracy

def _print_info(n_tokens, total_true_entities, total_predicted_entities, total_correct):
    print('processed {len} tokens ' \
          'with {tot_true} phrases; ' \
          'found: {tot_pred} phrases; ' \
          'correct: {tot_cor}.\n'.format(len=n_tokens,
                                         tot_true=total_true_entities,
                                         tot_pred=total_predicted_entities,
                                         tot_cor=total_correct))

def _print_metrics(accuracy, total_precision, total_recall, total_f1):
    print('precision:  {tot_prec:.2f}%; ' \
          'recall:  {tot_recall:.2f}%; ' \
          'F1:  {tot_f1:.2f}\n'.format(acc=accuracy,
                                           tot_prec=total_precision,
                                           tot_recall=total_recall,
                                           tot_f1=total_f1))

def _print_tag_metrics(tag, tag_results):
    print(('\t%12s' % tag) + ': precision:  {tot_prec:6.2f}%; ' \
                               'recall:  {tot_recall:6.2f}%; ' \
                               'F1:  {tot_f1:6.2f}; ' \
                               'predicted:  {tot_predicted:4d}\n'.format(tot_prec=tag_results['precision'],
                                                                         tot_recall=tag_results['recall'],
                                                                         tot_f1=tag_results['f1'],
                                                                         tot_predicted=tag_results['n_predicted_entities']))

class ScoreEvaluator:
	def __init__(self, token_to_idx, idx_to_tag, idx_to_token):
		self.token_to_idx = token_to_idx
		self.idx_to_tag = idx_to_tag
		self.idx_to_token = idx_to_token

	@staticmethod
	def precision_recall_f1(y_true, y_pred, print_results=True, short_report=False):
	    # Find all tagy_trues
	    tags = sorted(set(tag[2:] for tag in y_true + y_pred if tag != 'O'))

	    results = OrderedDict((tag, OrderedDict()) for tag in tags)
	    n_tokens = len(y_true)
	    total_correct = 0

	    # For eval_conll_try we find all chunks in the ground truth and prediction
	    # For each chunk we store starting and ending indices
	    for tag in tags:
	        true_chunk = list()
	        predicted_chunk = list()
	        for position in range(n_tokens):
	            _update_chunk(y_true[position], y_true[position - 1], tag, true_chunk, position)
	            _update_chunk(y_pred[position], y_pred[position - 1], tag, predicted_chunk, position, True)

	        _update_last_chunk(true_chunk, position)
	        _update_last_chunk(predicted_chunk, position)

	        # Then we find all correctly classified intervals
	        # True positive results
	        tp = sum(chunk in predicted_chunk for chunk in true_chunk)
	        total_correct += tp

	        # And then just calculate errors of the first and second kind
	        # False negative
	        fn = len(true_chunk) - tp
	        # False positive
	        fp = len(predicted_chunk) - tp
	        precision, recall, f1 = _tag_precision_recall_f1(tp, fp, fn)

	        results[tag]['precision'] = precision
	        results[tag]['recall'] = recall
	        results[tag]['f1'] = f1
	        results[tag]['n_predicted_entities'] = len(predicted_chunk)
	        results[tag]['n_true_entities'] = len(true_chunk)

	    total_true_entities, total_predicted_entities, \
	           total_precision, total_recall, total_f1, accuracy = _aggregate_metrics(results, total_correct)

	    if print_results:
	        _print_info(n_tokens, total_true_entities, total_predicted_entities, total_correct)
	        _print_metrics(accuracy, total_precision, total_recall, total_f1)

	        if not short_report:
	            for tag, tag_results in results.items():
	                _print_tag_metrics(tag, tag_results)
	    if short_report:
	    	results = {
	            'precision': total_precision,
	            'recall': total_recall,
	            'f1': total_f1,
	            'n_predicted_entities': total_predicted_entities,
	            'n_true_entities': total_true_entities,
	    	}
	    return results

	def predict_tags(self, model, token_idxs_batch):
	    """Performs predictions and transforms indices to tokens and tags."""
	    
	    tag_idxs_batch = model.predict_for_batch(token_idxs_batch)
	    tags_batch, tokens_batch = [], []
	    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
	        tags, tokens = [], []
	        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
	            if token_idx != self.token_to_idx['<PAD>']:
	                tags.append(self.idx_to_tag[tag_idx])
	                tokens.append(self.idx_to_token[token_idx])
	        tags_batch.append(tags)
	        tokens_batch.append(tokens)
	    return tags_batch, tokens_batch
	    
	def eval_conll(self, model, data_loader, print_results=False, short_report=True):
	    """Computes NER quality measures using CONLL shared task script."""
	    
	    y_true, y_pred = [], []
	    for x_batch, y_batch in data_loader:
	        pred_tags_batch, tokens_batch = self.predict_tags(model, x_batch)
	        ground_truth_tags = [
	        	[self.idx_to_tag[tag_idx] for tag_idx in true_tag_sequence][:len(pred_tag_sequence)]
	        	for true_tag_sequence, pred_tag_sequence in zip(y_batch, pred_tags_batch)
	        ]

	        # We extend every prediction and ground truth sequence with 'O' tag
	        # to indicate a possible end of entity.
	        for true_sequence, pred_sequence in zip(ground_truth_tags, pred_tags_batch):
	            y_true += true_sequence + ['O']
	            y_pred += pred_sequence + ['O']

	    results = self.precision_recall_f1(y_true, y_pred, print_results=print_results, short_report=short_report)
	    return results