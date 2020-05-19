import torch
import itertools
import pytorch_wrapper.functional as pwF

from torch.utils.data import Dataset

from ...utils.sequences import pad_to_max
from ..utils import parse_ner_dataset_file


class NERRNNDataset(Dataset):
    I2L = [
        'B-LOC',
        'B-ORG',
        'B-PER',
        'I-LOC',
        'I-ORG',
        'I-PER',
        'O'
    ]
    L2I = {k: i for i, k in enumerate(I2L)}

    def __init__(self, dataset_file, w2i, c2i):
        self.ids = []
        self.processed_tokens = []
        self.processed_tokens_len = []
        self.char_words = []
        self.char_word_lens = []
        self.targets = []

        for i, tokenlist in enumerate(parse_ner_dataset_file(dataset_file)):
            cur_words, cur_words_len, cur_char_words, cur_char_word_lens, cur_targets = \
                self.process_example(tokenlist, w2i, c2i)

            self.processed_tokens.append(cur_words)
            self.processed_tokens_len.append(cur_words_len)
            self.char_words.append(cur_char_words)
            self.char_word_lens.append(cur_char_word_lens)
            self.targets.append([self.L2I[t] for t in cur_targets])
            self.ids.append(i)

    def __getitem__(self, index):
        return (
            self.ids[index],
            (
                self.char_words[index],
                self.char_word_lens[index],
                self.processed_tokens[index],
                self.processed_tokens_len[index],
            ),
            self.targets[index]
        )

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch):
        batch_zipped = list(zip(*batch))
        input_zipped = list(zip(*batch_zipped[1]))

        ids = batch_zipped[0]

        batched_char_words = torch.tensor(
            pad_to_max(list(itertools.chain.from_iterable(input_zipped[0]))),
            dtype=torch.long
        )
        batched_char_words_len = torch.tensor(list(itertools.chain.from_iterable(input_zipped[1])), dtype=torch.int)

        nbs_accumulated = list(itertools.accumulate([1] + list(input_zipped[3])))
        indices = [list(range(nbs_accumulated[i], nbs_accumulated[i + 1])) for i in range(len(nbs_accumulated) - 1)]
        batched_char_word_index = torch.tensor(pad_to_max(indices), dtype=torch.long)

        batched_tokens = torch.tensor(pad_to_max(input_zipped[2]), dtype=torch.long)
        batched_tokens_len = torch.tensor(input_zipped[3], dtype=torch.int)

        with torch.no_grad():
            pred_mask = pwF.create_mask_from_length(batched_tokens_len, torch.max(batched_tokens_len).item())

        target = torch.tensor(pad_to_max(batch_zipped[2], pad_value=-1), dtype=torch.long)

        return {
            'id': ids,
            'input': [
                batched_char_words,
                batched_char_words_len,
                batched_char_word_index,
                batched_tokens,
                batched_tokens_len,
                target
            ],
            'target': target,
            'mask': pred_mask
        }

    @staticmethod
    def process_example(tokens, w2i, c2i):

        processed_tokens = []
        char_words = []
        char_word_lens = []
        targets = []

        for token in tokens:
            processed_tokens.append(w2i[token['text'].lower()])
            char_words.append(
                [c2i['<SOW>']] +
                [c2i.get(c, 1) for c in list(token['text'])] +
                [c2i['<EOW>']]
            )
            char_word_lens.append(len(char_words[-1]))
            targets.append(token['label'])

        return processed_tokens, len(processed_tokens), char_words, char_word_lens, targets
