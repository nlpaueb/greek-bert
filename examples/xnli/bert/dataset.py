import torch
import json

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from ...utils.sequences import pad_to_max


class XNLIBERTDataset(Dataset):
    L2I = {
        'neutral': 0,
        'contradiction': 1,
        'contradictory': 1,
        'entailment': 2
    }

    def __init__(self, file, tokenizer, preprocessing_function):
        self.ids = []
        self.texts = []
        self.texts_len = []
        self.targets = []

        for i, l in enumerate(tqdm(file)):
            ex = json.loads(l)
            cur_text, cur_len = self.process_example(
                ex,
                tokenizer,
                preprocessing_function
            )
            self.texts.append(cur_text)
            self.texts_len.append(cur_len)
            self.targets.append(self.L2I[ex['label']])
            self.ids.append(i)

    def __getitem__(self, index):
        return (
            self.ids[index],
            (self.texts[index], self.texts_len[index]),
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
        texts_len = torch.tensor(input_zipped[1], dtype=torch.int)

        target = torch.tensor(batch_zipped[2], dtype=torch.long)

        batch = {
            'id': ids,
            'input': [texts, texts_len],
            'target': target
        }

        return batch

    @staticmethod
    def process_example(ex, tokenizer, preprocessing_function):
        tokens = tokenizer.encode(
            preprocessing_function(ex['prem']) if preprocessing_function else ex['prem'],
            text_pair=preprocessing_function(ex['hypo']) if preprocessing_function else ex['hypo'],
            add_special_tokens=True,
            max_length=512
        )

        return tokens, len(tokens)
