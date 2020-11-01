import numpy as np

def test_encode(encode):
    result = encode('this is some text', 'this is another text')

    assert result['input_ids'] == [101, 2023, 2003, 2070, 3793, 102, 2023, 2003, 2178, 3793, 102], \
        'input_ids should be [101, 2023, 2003, 2070, 3793, 102, 2023, 2003, 2178, 3793, 102]'
    assert result['token_type_ids'] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], \
        'token_type_ids should be [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]'
    
def test_dataset(dataset):
    assert len(dataset[0]) == 3, 'Dataset[idx] should output tuple with 3 elements.'
    assert isinstance(dataset[0][-1], np.int64) or isinstance(dataset[0][-1], int), \
        'target should np.int64 or int'
    
def test_collator(dataset, collate_fn):
    ids, token_type_ids, labels = collate_fn([dataset[i] for i in range(10)])
    assert ids.shape[0] == labels.shape[0] == token_type_ids.shape[0], \
        'ids, token_type_ids, labels shoud have equal first dimension'
    assert ids.shape[1] == token_type_ids.shape[1], 'Incorrect shape of ids or token_type_ids'
    
def test_model(dataloader, model, device):
    input_ids, token_type_ids, _ = map(lambda x: x.to(device), next(iter(dataloader)))
    pred_shape = model(input_ids, token_type_ids).shape
    assert len(pred_shape) == 1 and pred_shape[0] == input_ids.shape[0], \
        f'Incorrect shape for the output of the model: {pred_shape} instead of {[input_ids.shape[0]]}'