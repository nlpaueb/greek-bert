import torch

from torch.utils.data import Dataset
from conllu import parse_incr

from ...utils.text import strip_accents_and_lowercase
from ...utils.sequences import pad_to_max


class UDBERTDataset(Dataset):
    I2L = [
        'ADJ',
        'ADP',
        'ADV',
        'AUX',
        'CCONJ',
        'DET',
        'NOUN',
        'NUM',
        'PART',
        'PRON',
        'PROPN',
        'PUNCT',
        'SCONJ',
        'SYM',
        'VERB',
        'X',
        '_'
    ]
    L2I = {k: i for i, k in enumerate(I2L)}

    def __init__(self, dataset_file, tokenizer):

        self.ids = []
        self.texts = []
        self.text_lens = []
        self.pred_masks = []
        self.targets = []

        for i, tokenlist in enumerate(parse_incr(dataset_file)):
            cur_texts, cur_text_lens, pred_mask, labels = self.process_example(
                tokenlist,
                tokenizer
            )
            self.texts.append(cur_texts)
            self.text_lens.append(cur_text_lens)
            self.pred_masks.append(pred_mask)
            self.targets.append([self.L2I.get(cur_l, -1) for cur_l in labels])
            self.ids.append(i)

    def __getitem__(self, index):
        return (
            self.ids[index],
            (self.texts[index], self.text_lens[index]),
            self.targets[index],
            self.pred_masks[index]
        )

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch):
        batch_zipped = list(zip(*batch))
        input_zipped = list(zip(*batch_zipped[1]))

        ids = batch_zipped[0]
        texts = torch.tensor(pad_to_max(input_zipped[0]), dtype=torch.long)
        text_lens = torch.tensor(input_zipped[1], dtype=torch.int)
        target = torch.tensor(pad_to_max(batch_zipped[2], pad_value=-1), dtype=torch.long)
        pred_mask = torch.tensor(pad_to_max(batch_zipped[3]), dtype=torch.bool)

        batch = {
            'id': ids,
            'input': [texts, text_lens],
            'target': target,
            'mask': pred_mask
        }

        return batch

    @staticmethod
    def process_example(tokens, tokenizer):
        bert_tokens = [101]
        pred_mask = [0]
        labels = ['PAD']
        for token in tokens:
            processed_token = strip_accents_and_lowercase(token['form'])
            current_tokens = tokenizer.encode(processed_token, add_special_tokens=False)
            if len(current_tokens) == 0:
                current_tokens = [tokenizer.unk_token_id]
            bert_tokens.extend(current_tokens)
            labels.extend([token['upostag']] + ['PAD'] * (len(current_tokens) - 1))
            pred_mask.extend([1] + [0] * (len(current_tokens) - 1))
        bert_tokens.append(102)
        pred_mask.append(0)
        labels.append('PAD')

        return bert_tokens, len(bert_tokens), pred_mask, labels
