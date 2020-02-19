import click
import os
import urllib.request
import random
import json
import torch
import pytorch_wrapper as pw
import pytorch_wrapper.functional as pwF
import pytorch_wrapper.evaluators as pw_evaluators
import unicodedata

from tqdm.auto import tqdm
from zipfile import ZipFile
from tqdm import tqdm
from itertools import product
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, AdamW
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset


@click.group()
def cli():
    pass


@cli.command()
@click.argument('datasets_folder_path', type=str, default='tmp')
def download_data(datasets_folder_path):
    os.makedirs(datasets_folder_path, exist_ok=True)
    urllib.request.urlretrieve(
        'https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip',
        filename=f'{datasets_folder_path}/XNLI-MT-1.0.zip'
    )
    urllib.request.urlretrieve(
        'https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip',
        filename=f'{datasets_folder_path}/XNLI-1.0.zip'
    )

    for file in ['XNLI-MT-1.0.zip', 'XNLI-1.0.zip']:
        with ZipFile(f'{datasets_folder_path}/{file}', 'r') as z:
            z.extractall(datasets_folder_path)

    data = []
    with open(f'{datasets_folder_path}/XNLI-MT-1.0/multinli/multinli.train.el.tsv') as fr:
        _ = next(fr)
        for l in fr:
            premise, hypo, label = l.split('\t')
            data.append({
                'prem': premise,
                'hypo': hypo,
                'label': label.strip('\n')
            })

    random.seed(0)
    random.shuffle(data)

    with open(f'{datasets_folder_path}/xnli_el_train.jsonl', 'w') as fw:
        for i in range(50000):
            fw.write(json.dumps(data[i]) + '\n')

    for eval_ds in ['dev', 'test']:
        with open(f'{datasets_folder_path}/XNLI-1.0/xnli.{eval_ds}.tsv') as fr:
            next(fr)
            with open(f'{datasets_folder_path}/xnli_el_{eval_ds}.jsonl', 'w') as fw:
                for l in fr:
                    ex = l.split('\t')
                    if ex[0] == 'el':
                        fw.write(json.dumps({'prem': ex[16], 'hypo': ex[17], 'label': ex[1]}) + '\n')


@cli.group()
def multi_bert():
    pass


@multi_bert.command()
@click.argument('datasets_folder_path', type=str, default='tmp')
@click.option('--multi-gpu', is_flag=True)
def tune(path, multi_gpu):
    results = tune_bert_model('bert-base-multilingual-uncased', path, multi_gpu)
    print(max(results, key=lambda x: x[0]))


@multi_bert.command()
@click.argument('datasets_folder_path', type=str, default='tmp')
@click.argument('lr', type=float)
@click.argument('dp', type=float)
@click.argument('grad_accumulation_steps', type=int)
@click.option('--multi-gpu', is_flag=True)
def run(datasets_folder_path, lr, dp, grad_accumulation_steps, multi_gpu):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

    train_dataset = BERTXNLIDataset(f'{datasets_folder_path}/xnli_el_train.jsonl', tokenizer, L2I)
    dev_dataset = BERTXNLIDataset(f'{datasets_folder_path}/xnli_el_dev.jsonl', tokenizer, L2I)
    test_dataset = BERTXNLIDataset(f'{datasets_folder_path}/xnli_el_test.jsonl', tokenizer, L2I)
    model = AutoModel.from_pretrained('bert-base-multilingual-uncased')

    results = run_bert_experiment(
        train_dataset,
        dev_dataset,
        test_dataset,
        model,
        lr,
        dp,
        grad_accumulation_steps,
        multi_gpu
    )

    print(results)


@cli.group()
def greek_bert():
    pass


@greek_bert.command()
@click.argument('datasets_folder_path', type=str, default='tmp')
@click.option('--multi-gpu', is_flag=True)
def tune(datasets_folder_path, multi_gpu):
    results = tune_bert_model('nlpaueb/bert-base-greek-uncased-v1', datasets_folder_path, multi_gpu)
    print(max(results, key=lambda x: x[0]))


@greek_bert.command()
@click.argument('datasets_folder_path', type=str, default='tmp')
@click.argument('lr', type=float)
@click.argument('dp', type=float)
@click.argument('grad_accumulation_steps', type=int)
@click.option('--multi-gpu', is_flag=True)
def run(datasets_folder_path, lr, dp, grad_accumulation_steps, multi_gpu):
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')

    train_dataset = BERTXNLIDataset(f'{datasets_folder_path}/xnli_el_train.jsonl', tokenizer, L2I)
    dev_dataset = BERTXNLIDataset(f'{datasets_folder_path}/xnli_el_dev.jsonl', tokenizer, L2I)
    test_dataset = BERTXNLIDataset(f'{datasets_folder_path}/xnli_el_test.jsonl', tokenizer, L2I)
    model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')

    results = run_bert_experiment(
        train_dataset,
        dev_dataset,
        test_dataset,
        model,
        lr,
        dp,
        grad_accumulation_steps,
        multi_gpu
    )

    print(results)


L2I = {
    'neutral': 0,
    'contradictory': 1,
    'contradiction': 1,
    'entailment': 2
}


class BERTXNLIDataset(Dataset):
    def __init__(self, filename, tokenizer, l2i):
        self.ids = []
        self.texts = []
        self.texts_len = []
        self.targets = []

        with open(filename, encoding='utf-8') as fr:
            for i, l in enumerate(tqdm(fr)):
                ex = json.loads(l)
                cur_text, cur_len = BERTXNLIDataset.process_example(
                    ex,
                    tokenizer
                )
                self.texts.append(cur_text)
                self.texts_len.append(cur_len)
                self.targets.append(l2i[ex['label']])
                self.ids.append(f'{filename}-{i}')

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
        texts = torch.tensor(BERTXNLIDataset.pad_to_max(input_zipped[0]), dtype=torch.long)
        texts_len = torch.tensor(input_zipped[1], dtype=torch.int)

        target = torch.tensor(batch_zipped[2], dtype=torch.long)

        batch = {
            'id': ids,
            'input': [texts, texts_len],
            'target': target
        }

        return batch

    @staticmethod
    def process_example(ex, tokenizer):
        tokens = tokenizer.encode(
            BERTXNLIDataset.strip_accents_and_lowercase(ex['prem']),
            text_pair=BERTXNLIDataset.strip_accents_and_lowercase(ex['hypo']),
            add_special_tokens=True,
            max_length=512
        )

        return tokens, len(tokens)

    @staticmethod
    def strip_accents_and_lowercase(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

    @staticmethod
    def pad_to_max(lst, max_len=None, pad_value=0):
        pad = len(max(lst, key=len))
        if max_len is not None:
            pad = min(max_len, pad)

        return [i + [pad_value for _ in range(pad - len(i))] if len(i) <= pad else i[:pad] for i in lst]


class BERTClassificationModel(nn.Module):

    def __init__(self, model, dp):
        super(BERTClassificationModel, self).__init__()
        self.model = model
        self.dp = nn.Dropout(dp)
        self.output_linear = nn.Linear(768, 3)

    def forward(self, text, text_len):
        attention_mask = pwF.create_mask_from_length(text_len, text.shape[1])
        return self.output_linear(self.dp(self.model(text, attention_mask=attention_mask)[0][:, 0, :]))


def run_bert_experiment(train_dataset,
                        dev_dataset,
                        eval_dataset,
                        model, lr, dp,
                        grad_accumulation_steps,
                        run_on_multi_gpus):
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=16,
        collate_fn=BERTXNLIDataset.collate_fn
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        sampler=SequentialSampler(dev_dataset),
        batch_size=16,
        collate_fn=BERTXNLIDataset.collate_fn
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=16,
        collate_fn=BERTXNLIDataset.collate_fn
    )

    evaluators = {

        'acc': pw_evaluators.MultiClassAccuracyEvaluator(),
        'macro-prec': pw_evaluators.MultiClassPrecisionEvaluator(average='macro'),
        'macro-rec': pw_evaluators.MultiClassRecallEvaluator(average='macro'),
        'macro-f1': pw_evaluators.MultiClassF1Evaluator(average='macro')
    }

    model = BERTClassificationModel(model, dp)
    if torch.cuda.is_available():
        system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cuda'))
    else:
        system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cpu'))

    loss_wrapper = pw.loss_wrappers.GenericPointWiseLossWrapper(nn.CrossEntropyLoss())
    optimizer = AdamW(model.parameters(), lr=lr)

    os.makedirs('tmp/', exist_ok=True)

    train_method = system.train_on_multi_gpus if run_on_multi_gpus else system.train

    _ = train_method(
        loss_wrapper,
        optimizer,
        train_data_loader=train_dataloader,
        evaluation_data_loaders={'dev': dev_dataloader},
        evaluators=evaluators,
        callbacks=[
            pw.training_callbacks.EarlyStoppingCriterionCallback(
                patience=3,
                gradient_accumulation_steps=grad_accumulation_steps,
                evaluation_data_loader_key='dev',
                evaluator_key='macro-f1',
                tmp_best_state_filepath=f'tmp/temp.es.weights'
            )
        ]
    )

    return system.evaluate(eval_dataloader, evaluators)


def tune_bert_model(pretrained_bert_name, datasets_folder_path, run_on_multi_gpus):
    lrs = [5e-5, 3e-5, 2e-5]
    dp = [0, 0.1, 0.2]
    grad_accumulation_steps = [1, 2]
    params = list(product(lrs, dp, grad_accumulation_steps))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)

    train_dataset = BERTXNLIDataset(f'{datasets_folder_path}/xnli_el_train.jsonl', tokenizer, L2I)
    dev_dataset = BERTXNLIDataset(f'{datasets_folder_path}/xnli_el_dev.jsonl', tokenizer, L2I)

    results = []
    for i, (lr, dp, grad_accumulation_steps) in enumerate(params):
        print(f'{i + 1}/{len(params)}')
        model = AutoModel.from_pretrained(pretrained_bert_name)
        current_results = run_bert_experiment(
            train_dataset,
            dev_dataset,
            dev_dataset,
            model,
            lr,
            dp,
            grad_accumulation_steps,
            run_on_multi_gpus
        )
        results.append([current_results['macro-f1'].score, (lr, dp, grad_accumulation_steps)])

    return results


if __name__ == '__main__':
    cli()
