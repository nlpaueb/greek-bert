import click
import os
import json
import fasttext
import numpy as np
import pickle

from zipfile import ZipFile

from .bert.system_wrapper import XNLIBERTSystemWrapper
from .rnn.system_wrapper import XNLIRNNSystemWrapper
from .rnn.dataset import XNLIRNNDataset
from ..utils.fasttext_downloader import download_model


@click.group()
def xnli():
    pass


@xnli.command()
@click.argument('data_folder_path', type=str, default='data')
def download_data(data_folder_path):
    os.makedirs(data_folder_path, exist_ok=True)

    os.system(
        f'wget http://nlp.cs.aueb.gr/software_and_datasets/xnli_el.zip -P {data_folder_path}'
    )

    with ZipFile(f'{data_folder_path}/xnli_el.zip', 'r') as z:
        z.extractall(data_folder_path)


@xnli.group()
def multi_bert():
    pass


@multi_bert.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--multi-gpu', is_flag=True)
def tune(data_folder_path, multi_gpu):
    train_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.train.jsonl')
    val_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.val.jsonl')

    results = XNLIBERTSystemWrapper.tune(
        'bert-base-multilingual-uncased',
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@multi_bert.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=2e-05)
@click.option('--dp', type=float, default=0.)
@click.option('--grad-accumulation-steps', type=int, default=2)
@click.option('--multi-gpu', is_flag=True)
def run(data_folder_path, batch_size, lr, dp, grad_accumulation_steps, multi_gpu):
    sw = XNLIBERTSystemWrapper('bert-base-multilingual-uncased', {'dp': dp})

    train_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.train.jsonl')
    val_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.val.jsonl')
    test_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.test.jsonl')

    sw.train(train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


@xnli.group()
def greek_bert():
    pass


@greek_bert.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--multi-gpu', is_flag=True)
def tune(data_folder_path, multi_gpu):
    train_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.train.jsonl')
    val_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.val.jsonl')

    results = XNLIBERTSystemWrapper.tune(
        'nlpaueb/bert-base-greek-uncased-v1',
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@greek_bert.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=2e-05)
@click.option('--dp', type=float, default=0.)
@click.option('--grad-accumulation-steps', type=int, default=2)
@click.option('--multi-gpu', is_flag=True)
def run(data_folder_path, batch_size, lr, dp, grad_accumulation_steps, multi_gpu):
    sw = XNLIBERTSystemWrapper('nlpaueb/bert-base-greek-uncased-v1', {'dp': dp})

    train_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.train.jsonl')
    val_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.val.jsonl')
    test_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.test.jsonl')

    sw.train(train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


@xnli.group()
def rnn():
    pass


@rnn.command()
@click.argument('data_folder_path', type=str, default='data')
def download_embeddings(data_folder_path):
    download_model('el', data_folder_path, if_exists='ignore')
    ft = fasttext.load_model(f'{data_folder_path}/cc.el.300.bin')

    vocab = set()
    for ds_name in ['train', 'val', 'test']:
        with open(f'{data_folder_path}/xnli_el/xnli.el.{ds_name}.jsonl') as fr:
            for line in fr:
                ex = json.loads(line)
                vocab.update(XNLIRNNDataset.process_text(ex['prem']).split())
                vocab.update(XNLIRNNDataset.process_text(ex['hypo']).split())

    word_vectors = []
    i2w = list(vocab)
    for word in i2w:
        word_vectors.append(ft.get_word_vector(word))
    word_vectors = [[0] * len(word_vectors[0])] + word_vectors
    i2w = ['<PAD>'] + i2w
    w2i = {w: i for i, w in enumerate(i2w)}

    with open(f'{data_folder_path}/xnli_el/xnli_ft.pkl', 'wb') as fw:
        pickle.dump((np.array(word_vectors), w2i, i2w), fw)


@rnn.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--multi-gpu', is_flag=True)
def tune(data_folder_path, multi_gpu):
    train_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.train.jsonl')
    val_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.val.jsonl')

    with open(f'{data_folder_path}/xnli_el/xnli_ft.pkl', 'rb') as fr:
        embeddings, w2i, _ = pickle.load(fr)

    results = XNLIRNNSystemWrapper.tune(
        embeddings,
        w2i,
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@rnn.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--batch-size', type=int, default=64)
@click.option('--lr', type=float, default=1e-03)
@click.option('--dp', type=float, default=0.2)
@click.option('--rnn-hs', type=int, default=100)
@click.option('--grad-accumulation-steps', type=int, default=1)
@click.option('--multi-gpu', is_flag=True)
def run(data_folder_path, batch_size, lr, dp, rnn_hs, grad_accumulation_steps, multi_gpu):
    with open(f'{data_folder_path}/xnli_el/xnli_ft.pkl', 'rb') as fr:
        embeddings, w2i, _ = pickle.load(fr)

    sw = XNLIRNNSystemWrapper(
        embeddings,
        w2i, {
            'rnn_dp': dp,
            'mlp_dp': dp,
            'rnn_hidden_size': rnn_hs,
        })

    train_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.train.jsonl')
    val_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.val.jsonl')
    test_dataset_file = open(f'{data_folder_path}/xnli_el/xnli.el.test.jsonl')

    sw.train(train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


if __name__ == '__main__':
    xnli()
