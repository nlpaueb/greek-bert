import click
import fasttext
import numpy as np
import pickle

from ..utils.fasttext_downloader import download_model
from .utils import parse_ner_dataset_file
from .bert.system_wrapper import NERBERTSystemWrapper
from .rnn.system_wrapper import NERRNNSystemWrapper


@click.group()
def ner():
    pass


@ner.group()
def multi_bert():
    pass


@multi_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ner/train.txt')
@click.argument('val_dataset_file', type=click.File('r'), default='data/ner/dev.txt')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, val_dataset_file, multi_gpu):
    results = NERBERTSystemWrapper.tune(
        'bert-base-multilingual-uncased',
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@multi_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ner/train.txt')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ner/dev.txt')
@click.argument('test_dataset_file', type=click.File('r'), default='data/ner/test.txt')
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=3e-05)
@click.option('--dp', type=float, default=0.2)
@click.option('--grad-accumulation-steps', type=int, default=4)
@click.option('--multi-gpu', is_flag=True)
def run(train_dataset_file, dev_dataset_file, test_dataset_file, batch_size, lr, dp, grad_accumulation_steps,
        multi_gpu):
    sw = NERBERTSystemWrapper('bert-base-multilingual-uncased', {'dp': dp})

    sw.train(train_dataset_file, dev_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


@ner.group()
def greek_bert():
    pass


@greek_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ner/train.txt')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ner/dev.txt')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, dev_dataset_file, multi_gpu):
    results = NERBERTSystemWrapper.tune(
        'nlpaueb/bert-base-greek-uncased-v1',
        train_dataset_file,
        dev_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@greek_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ner/train.txt')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ner/dev.txt')
@click.argument('test_dataset_file', type=click.File('r'), default='data/ner/test.txt')
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=5e-05)
@click.option('--dp', type=float, default=0.1)
@click.option('--grad-accumulation-steps', type=int, default=2)
@click.option('--multi-gpu', is_flag=True)
def run(train_dataset_file, dev_dataset_file, test_dataset_file, batch_size, lr, dp, grad_accumulation_steps,
        multi_gpu):
    sw = NERBERTSystemWrapper('nlpaueb/bert-base-greek-uncased-v1', {'dp': dp})

    sw.train(train_dataset_file, dev_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


@ner.group()
def rnn():
    pass


@rnn.command()
@click.argument('tmp_download_path', type=str, default='data')
@click.argument('embeddings_save_path', type=str, default='data/ner/ner_ft.pkl')
@click.argument('dataset_file_paths', type=str, nargs=-1)
def download_embeddings(tmp_download_path, embeddings_save_path, dataset_file_paths):
    download_model('el', tmp_download_path, if_exists='ignore')
    ft = fasttext.load_model(f'{tmp_download_path}/cc.el.300.bin')

    if not dataset_file_paths:
        dataset_file_paths = [f'data/ner/{ds}.txt' for ds in ('train', 'dev', 'test', 'silver_train')]

    vocab = set()
    for p in dataset_file_paths:
        with open(p) as fr:
            for e in parse_ner_dataset_file(fr):
                for t in e:
                    vocab.add(t['text'].lower())

    word_vectors = []
    i2w = list(vocab)
    for word in i2w:
        word_vectors.append(ft.get_word_vector(word))
    word_vectors = [[0] * len(word_vectors[0])] + word_vectors
    i2w = ['<PAD>'] + i2w
    w2i = {w: i for i, w in enumerate(i2w)}

    with open(embeddings_save_path, 'wb') as fw:
        pickle.dump((np.array(word_vectors), w2i, i2w), fw)


@rnn.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ner/train.txt')
@click.argument('char_vocab_save_path', type=str, default='data/ner/char_voc.pkl')
def create_char_vocab(train_dataset_file, char_vocab_save_path):

    vocab = set()
    for e in parse_ner_dataset_file(train_dataset_file):
        for t in e:
            vocab.update(list(t['text']))

    c2i = {c: i + 4 for i, c in enumerate(vocab)}
    c2i['<PAD>'] = 0
    c2i['<UNK>'] = 1
    c2i['<SOW>'] = 2
    c2i['<EOW>'] = 3

    with open(char_vocab_save_path, 'wb') as fw:
        pickle.dump(c2i, fw)


@rnn.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ner/train.txt')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ner/dev.txt')
@click.argument('embeddings_file', type=click.File('rb'), default='data/ner/ner_ft.pkl')
@click.argument('char_vocab_file', type=click.File('rb'), default='data/ner/char_voc.pkl')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, dev_dataset_file, embeddings_file, char_vocab_file, multi_gpu):
    embeddings, w2i, _ = pickle.load(embeddings_file)
    c2i = pickle.load(char_vocab_file)

    results = NERRNNSystemWrapper.tune(
        embeddings,
        w2i,
        c2i,
        train_dataset_file,
        dev_dataset_file,
        multi_gpu
    )

    click.echo(max(results, key=lambda x: x[0]))


@rnn.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ner/train.txt')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ner/dev.txt')
@click.argument('test_dataset_file', type=click.File('r'), default='data/ner/test.txt')
@click.argument('embeddings_file', type=click.File('rb'), default='data/ner/ner_ft.pkl')
@click.argument('char_vocab_file', type=click.File('rb'), default='data/ner/char_voc.pkl')
@click.option('--batch-size', type=int, default=16)
@click.option('--lr', type=float, default=1e-03)
@click.option('--dp', type=float, default=0.3)
@click.option('--rnn-hs', type=int, default=100)
@click.option('--char-emb-size', type=int, default=30)
@click.option('--grad-accumulation-steps', type=int, default=1)
@click.option('--multi-gpu', is_flag=True)
def run(train_dataset_file, dev_dataset_file, test_dataset_file, embeddings_file, char_vocab_file, batch_size, lr, dp,
        rnn_hs, char_emb_size, grad_accumulation_steps, multi_gpu):
    embeddings, w2i, _ = pickle.load(embeddings_file)
    c2i = pickle.load(char_vocab_file)

    sw = NERRNNSystemWrapper(
        embeddings,
        w2i,
        c2i,
        {
            'rnn_dp': dp,
            'mlp_dp': dp,
            'rnn_hidden_size': rnn_hs,
            'char_embeddings_shape': (len(c2i), char_emb_size)
        }
    )

    sw.train(train_dataset_file, dev_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu)

    click.echo(results)


if __name__ == '__main__':
    ner()
