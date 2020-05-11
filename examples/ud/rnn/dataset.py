import torch

from torch.utils.data import Dataset
from conllu import parse_incr

from ...utils.sequences import pad_to_max


class UDRNNDataset(Dataset):
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

    def __init__(self, dataset_file, w2i):
        self.ids = []
        self.texts = []
        self.text_lens = []
        self.targets = []

        for i, tokenlist in enumerate(parse_incr(dataset_file)):
            cur_texts, cur_text_lens, cur_targets = self.process_example(tokenlist, w2i)
            self.texts.append(cur_texts)
            self.text_lens.append(cur_text_lens)
            self.targets.append([self.L2I[t] for t in cur_targets])
            self.ids.append(i)

    def __getitem__(self, index):
        return (
            self.ids[index],
            (self.texts[index], self.text_lens[index]),
            self.targets[index]
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

        batch = {
            'id': ids,
            'input': [texts, text_lens],
            'target': target
        }

        return batch

    @staticmethod
    def process_example(tokens, w2i):
        text = [w2i[token['form'].lower()] for token in tokens]
        targets = [token['upostag'] for token in tokens]
        return text, len(text), targets
