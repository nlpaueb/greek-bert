import click
import os
import fasttext
import numpy as np
import pickle
import urllib.request

from conllu import parse_incr

from ..utils.fasttext_downloader import download_model
from .bert.system_wrapper import UDBERTSystemWrapper
from .rnn.system_wrapper import UDRNNSystemWrapper


@click.group()
def ud():
    pass


@ud.command()
@click.argument('data_folder_path', type=str, default='data')
def download_data(data_folder_path):
    os.makedirs(f'{data_folder_path}/ud/', exist_ok=True)
    for name, url in [
        ('train', 'https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/master/el_gdt-ud-train.conllu'),
        ('dev', 'https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/master/el_gdt-ud-dev.conllu'),
        ('test', 'https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/master/el_gdt-ud-test.conllu')
    ]:
        urllib.request.urlretrieve(url, f'{data_folder_path}/ud/{name}.conllu')


@ud.group()
def multi_bert():
    pass


@multi_bert.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--multi-gpu', is_flag=True)
def tune(data_folder_path, multi_gpu):
    train_dataset_file = open(f'{data_folder_path}/ud/train.conllu')
    val_dataset_file = open(f'{data_folder_path}/ud/dev.conllu')

    results = UDBERTSystemWrapper.tune(
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
    sw = UDBERTSystemWrapper('bert-base-multilingual-uncased', {'dp': dp})

    train_dataset_file = open(f'{data_folder_path}/ud/train.conllu')
    val_dataset_file = open(f'{data_folder_path}/ud/dev.conllu')
    test_dataset_file = open(f'{data_folder_path}/ud/test.conllu')

    sw.train(train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


@ud.group()
def greek_bert():
    pass


@greek_bert.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--multi-gpu', is_flag=True)
def tune(data_folder_path, multi_gpu):
    train_dataset_file = open(f'{data_folder_path}/ud/train.conllu')
    val_dataset_file = open(f'{data_folder_path}/ud/dev.conllu')

    results = UDBERTSystemWrapper.tune(
        'nlpaueb/bert-base-greek-uncased-v1',
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@greek_bert.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=3e-05)
@click.option('--dp', type=float, default=0.2)
@click.option('--grad-accumulation-steps', type=int, default=4)
@click.option('--multi-gpu', is_flag=True)
def run(data_folder_path, batch_size, lr, dp, grad_accumulation_steps, multi_gpu):
    sw = UDBERTSystemWrapper('nlpaueb/bert-base-greek-uncased-v1', {'dp': dp})

    train_dataset_file = open(f'{data_folder_path}/ud/train.conllu')
    val_dataset_file = open(f'{data_folder_path}/ud/dev.conllu')
    test_dataset_file = open(f'{data_folder_path}/ud/test.conllu')

    sw.train(train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


@ud.group()
def rnn():
    pass


@rnn.command()
@click.argument('data_folder_path', type=str, default='data')
def download_embeddings(data_folder_path):
    download_model('el', data_folder_path, if_exists='ignore')
    ft = fasttext.load_model(f'{data_folder_path}/cc.el.300.bin')

    vocab = set()
    for ds_name in ['train', 'dev', 'test']:
        with open(f'{data_folder_path}/ud/{ds_name}.conllu') as fr:
            for e in parse_incr(fr):
                for t in e:
                    vocab.add(t['form'].lower())

    word_vectors = []
    i2w = list(vocab)
    for word in i2w:
        word_vectors.append(ft.get_word_vector(word))
    word_vectors = [[0] * len(word_vectors[0])] + word_vectors
    i2w = ['<PAD>'] + i2w
    w2i = {w: i for i, w in enumerate(i2w)}

    with open(f'{data_folder_path}/ud/ud_ft.pkl', 'wb') as fw:
        pickle.dump((np.array(word_vectors), w2i, i2w), fw)


@rnn.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--multi-gpu', is_flag=True)
def tune(data_folder_path, multi_gpu):
    train_dataset_file = open(f'{data_folder_path}/ud/train.conllu')
    val_dataset_file = open(f'{data_folder_path}/ud/dev.conllu')

    with open(f'{data_folder_path}/ud/ud_ft.pkl', 'rb') as fr:
        embeddings, w2i, _ = pickle.load(fr)

    results = UDRNNSystemWrapper.tune(
        embeddings,
        w2i,
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@rnn.command()
@click.argument('data_folder_path', type=str, default='data')
@click.option('--batch-size', type=int, default=16)
@click.option('--lr', type=float, default=1e-03)
@click.option('--dp', type=float, default=0.1)
@click.option('--rnn-hs', type=int, default=100)
@click.option('--grad-accumulation-steps', type=int, default=1)
@click.option('--multi-gpu', is_flag=True)
def run(data_folder_path, batch_size, lr, dp, rnn_hs, grad_accumulation_steps, multi_gpu):
    with open(f'{data_folder_path}/ud/ud_ft.pkl', 'rb') as fr:
        embeddings, w2i, _ = pickle.load(fr)

    sw = UDRNNSystemWrapper(
        embeddings,
        w2i, {
            'rnn_dp': dp,
            'mlp_dp': dp,
            'rnn_hidden_size': rnn_hs,
        })

    train_dataset_file = open(f'{data_folder_path}/ud/train.conllu')
    val_dataset_file = open(f'{data_folder_path}/ud/dev.conllu')
    test_dataset_file = open(f'{data_folder_path}/ud/test.conllu')

    sw.train(train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


if __name__ == '__main__':
    ud()
