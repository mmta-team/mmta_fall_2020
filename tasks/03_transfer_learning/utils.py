import numpy as np
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class InferenceDataset(Dataset):
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = torch.tensor(self.data[idx]['input_ids'], dtype=torch.long)
        type_ids = torch.tensor(self.data[idx]['token_type_ids'], dtype=torch.long)
        return ids, type_ids

    
def collate_fn(batch, pad_idx=0):
    ids, type_ids = map(lambda x: pad_sequence(x, batch_first=True, padding_value=pad_idx), zip(*batch))
    return ids, type_ids


def hits_count(dup_ranks, k):
    ranks = [rank <= k for rank in dup_ranks]
    return 0. if not ranks else np.mean(ranks)


def dcg_score(dup_ranks, k):
    vals = [1. / np.log2(1. + rank) for rank in dup_ranks if rank <= k]
    return 0. if not vals else np.sum(vals) / len(dup_ranks)


class Evaluator:

    def __init__(self, path, tokenizer, maxlen, batch_size, pad_idx=0, verbose=False):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.pad_idx = pad_idx
        
        data = []
        for line in open(path, encoding='utf-8'):
            data.append(line.strip().split('\t'))

        lengths = []
        prep_data = []
        for query, *docs in tqdm.tqdm(data, disable=not verbose, desc='Encoding text...'):
            for doc in docs:
                prep_data.append(self.encode(query, doc))
            lengths.append(len(docs))
        self.bounds = np.cumsum([0] + lengths)
        self.ids, prep_data = zip(*sorted(enumerate(prep_data), key=lambda x: len(x[1]['input_ids'])))

        ds = InferenceDataset(prep_data)
        self.dataloader = DataLoader(ds, batch_size, collate_fn=collate_fn)
        
    def __call__(self, model, device, verbose=False):
        model.to(device)
        model.eval()
        
        preds = []
        for batch in tqdm.tqdm(self.dataloader, disable=not verbose, desc='Computing predictions...'):
            input_ids, token_type_ids = map(lambda x: x.to(device), batch)
            attention_mask = input_ids != self.pad_idx
            with torch.no_grad():
                pred = model(input_ids, attention_mask, token_type_ids).cpu()
            preds.append(pred)
        preds = torch.cat(preds).numpy()
        
        _, preds = zip(*sorted(zip(self.ids, preds), key=lambda x: x[0]))
        
        rankings = []
        for i in range(len(self.bounds) - 1):
            rankings.append(
                list(np.argsort(-np.array(preds[self.bounds[i]:self.bounds[i + 1]]))).index(0) + 1
            )
            
        metrics = {
            'DCG': {f'DCG@{k}': dcg_score(rankings, k) for k in [1, 5, 10, 100, 500, 1000]},
            'Hits': {f'Hits@{k}': hits_count(rankings, k) for k in [1, 5, 10, 100, 500, 1000]}
        }
        
        return metrics
            
    def encode(self, query, doc):
        enc = self.tokenizer.encode_plus(query, doc, add_special_tokens=True)
        return {'input_ids': enc.input_ids[:self.maxlen], 'token_type_ids': enc.token_type_ids[:self.maxlen]}