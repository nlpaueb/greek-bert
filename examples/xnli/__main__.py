import click
import os
import json
import fasttext
import numpy as np
import pickle

from zipfile import ZipFile


@click.group()
def xnli():
    pass


@xnli.command()
@click.argument('data_folder_path', type=str, default='data')
def download_data(data_folder_path):
    os.makedirs(data_folder_path, exist_ok=True)
    os.system(f'wget http://nlp.cs.aueb.gr/software_and_datasets/xnli_el.zip -P {data_folder_path}')

    with ZipFile(f'{data_folder_path}/xnli_el.zip', 'r') as z:
        z.extractall(data_folder_path)


@xnli.group()
def multi_bert():
    pass


@multi_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.train.jsonl')
@click.argument('val_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.val.jsonl')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, val_dataset_file, multi_gpu):
    from .bert.system_wrapper import XNLIBERTSystemWrapper

    results = XNLIBERTSystemWrapper.tune(
        'bert-base-multilingual-uncased',
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@multi_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.train.jsonl')
@click.argument('val_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.val.jsonl')
@click.argument('test_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.test.jsonl')
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=2e-05)
@click.option('--dp', type=float, default=0.2)
@click.option('--grad-accumulation-steps', type=int, default=4)
@click.option('--multi-gpu', is_flag=True)
def run(train_dataset_file, val_dataset_file, test_dataset_file, batch_size, lr, dp, grad_accumulation_steps,
        multi_gpu):
    from .bert.system_wrapper import XNLIBERTSystemWrapper

    sw = XNLIBERTSystemWrapper('bert-base-multilingual-uncased', {'dp': dp})

    sw.train(train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


@xnli.group()
def greek_bert():
    pass


@greek_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.train.jsonl')
@click.argument('val_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.val.jsonl')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, val_dataset_file, multi_gpu):
    from .bert.system_wrapper import XNLIBERTSystemWrapper

    results = XNLIBERTSystemWrapper.tune(
        'nlpaueb/bert-base-greek-uncased-v1',
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@greek_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.train.jsonl')
@click.argument('val_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.val.jsonl')
@click.argument('test_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.test.jsonl')
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=3e-05)
@click.option('--dp', type=float, default=0.2)
@click.option('--grad-accumulation-steps', type=int, default=4)
@click.option('--multi-gpu', is_flag=True)
def run(train_dataset_file, val_dataset_file, test_dataset_file, batch_size, lr, dp, grad_accumulation_steps,
        multi_gpu):
    from .bert.system_wrapper import XNLIBERTSystemWrapper

    sw = XNLIBERTSystemWrapper('nlpaueb/bert-base-greek-uncased-v1', {'dp': dp})

    sw.train(train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


@xnli.group()
def dam():
    pass


@dam.command()
@click.argument('tmp_download_path', type=str, default='data')
@click.argument('embeddings_save_path', type=str, default='data/xnli_el/xnli_ft.pkl')
@click.argument('dataset_file_paths', type=str, nargs=-1)
def download_embeddings(tmp_download_path, embeddings_save_path, dataset_file_paths):
    from ..utils.fasttext_downloader import download_model
    from .dam.dataset import XNLIDAMDataset

    # todo: add big train
    if not dataset_file_paths:
        dataset_file_paths = [f'data/xnli_el/xnli.el.{ds}.jsonl' for ds in ('train', 'val', 'test')]

    download_model('el', tmp_download_path, if_exists='ignore')
    ft = fasttext.load_model(f'{tmp_download_path}/cc.el.300.bin')

    vocab = set()
    for ds in dataset_file_paths:
        with open(ds) as fr:
            for line in fr:
                ex = json.loads(line)
                vocab.update(XNLIDAMDataset.process_text(ex['prem']))
                vocab.update(XNLIDAMDataset.process_text(ex['hypo']))

    word_vectors = []
    i2w = list(vocab)
    for word in i2w:
        word_vectors.append(ft.get_word_vector(word))
    word_vectors = [[0] * len(word_vectors[0])] + word_vectors
    i2w = ['<PAD>'] + i2w
    w2i = {w: i for i, w in enumerate(i2w)}

    with open(embeddings_save_path, 'wb') as fw:
        pickle.dump((np.array(word_vectors), w2i, i2w), fw)


@dam.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.train.jsonl')
@click.argument('val_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.val.jsonl')
@click.argument('embeddings_file', type=click.File('rb'), default='data/xnli_el/xnli_ft.pkl')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, val_dataset_file, embeddings_file, multi_gpu):
    from .dam.system_wrapper import XNLIDAMSystemWrapper

    embeddings, w2i, _ = pickle.load(embeddings_file)

    results = XNLIDAMSystemWrapper.tune(
        embeddings,
        w2i,
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@dam.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.train.jsonl')
@click.argument('val_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.val.jsonl')
@click.argument('test_dataset_file', type=click.File('r'), default='data/xnli_el/xnli.el.test.jsonl')
@click.argument('embeddings_file', type=click.File('rb'), default='data/xnli_el/xnli_ft.pkl')
@click.option('--batch-size', type=int, default=64)
@click.option('--lr', type=float, default=0.001)
@click.option('--dp', type=float, default=0.3)
@click.option('--grad-accumulation-steps', type=int, default=1)
@click.option('--multi-gpu', is_flag=True)
def run(train_dataset_file, val_dataset_file, test_dataset_file, embeddings_file, batch_size, lr, dp,
        grad_accumulation_steps, multi_gpu):
    from .dam.system_wrapper import XNLIDAMSystemWrapper

    embeddings, w2i, _ = pickle.load(embeddings_file)

    sw = XNLIDAMSystemWrapper(
        embeddings,
        w2i, {
            'mlp_dp': dp
        })

    sw.train(train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


if __name__ == '__main__':
    xnli()
