import click
import os
import fasttext
import numpy as np
import pickle
import urllib.request

from conllu import parse_incr

from ..utils.fasttext_downloader import download_model
from ..utils.text import strip_accents_and_lowercase
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
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('val_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, val_dataset_file, multi_gpu):
    results = UDBERTSystemWrapper.tune(
        'bert-base-multilingual-uncased',
        strip_accents_and_lowercase,
        True,
        train_dataset_file,
        val_dataset_file,
        multi_gpu
    )

    print(max(results, key=lambda x: x[0]))


@multi_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.argument('test_dataset_file', type=click.File('r'), default='data/ud/test.conllu')
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=2e-05)
@click.option('--dp', type=float, default=0.2)
@click.option('--grad-accumulation-steps', type=int, default=2)
@click.option('--multi-gpu', is_flag=True)
@click.option('--silent', is_flag=True)
@click.option('--seed', type=int, default=0)
def run(train_dataset_file, dev_dataset_file, test_dataset_file, batch_size, lr, dp, grad_accumulation_steps,
        multi_gpu, silent, seed):
    sw = UDBERTSystemWrapper(
        'bert-base-multilingual-uncased',
        strip_accents_and_lowercase,
        True,
        {'dp': dp}
    )

    sw.train(train_dataset_file, dev_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu, not silent, seed)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu, not silent)

    print(results)


@ud.group()
def greek_bert():
    pass


@greek_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, dev_dataset_file, multi_gpu):
    results = UDBERTSystemWrapper.tune(
        'nlpaueb/bert-base-greek-uncased-v1',
        strip_accents_and_lowercase,
        True,
        train_dataset_file,
        dev_dataset_file,
        multi_gpu
    )

    print(max(results, key=lambda x: x[0]))


@greek_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.argument('test_dataset_file', type=click.File('r'), default='data/ud/test.conllu')
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=5e-05)
@click.option('--dp', type=float, default=0.1)
@click.option('--grad-accumulation-steps', type=int, default=2)
@click.option('--multi-gpu', is_flag=True)
@click.option('--silent', is_flag=True)
@click.option('--seed', type=int, default=0)
def run(train_dataset_file, dev_dataset_file, test_dataset_file, batch_size, lr, dp, grad_accumulation_steps,
        multi_gpu, silent, seed):
    sw = UDBERTSystemWrapper(
        'nlpaueb/bert-base-greek-uncased-v1',
        strip_accents_and_lowercase,
        True,
        {'dp': dp}
    )

    sw.train(train_dataset_file, dev_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu, not silent, seed)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu, not silent)

    print(results)


@ud.group()
def cased_multi_bert():
    pass


@cased_multi_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, dev_dataset_file, multi_gpu):
    results = UDBERTSystemWrapper.tune(
        'bert-base-multilingual-cased',
        None,
        True,
        train_dataset_file,
        dev_dataset_file,
        multi_gpu
    )

    print(max(results, key=lambda x: x[0]))


@cased_multi_bert.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.argument('test_dataset_file', type=click.File('r'), default='data/ud/test.conllu')
@click.option('--batch-size', type=int, default=4)
@click.option('--lr', type=float, default=2e-05)
@click.option('--dp', type=float, default=0)
@click.option('--grad-accumulation-steps', type=int, default=8)
@click.option('--multi-gpu', is_flag=True)
@click.option('--silent', is_flag=True)
@click.option('--seed', type=int, default=0)
def run(train_dataset_file, dev_dataset_file, test_dataset_file, batch_size, lr, dp, grad_accumulation_steps,
        multi_gpu, silent, seed):
    sw = UDBERTSystemWrapper(
        'bert-base-multilingual-cased',
        None,
        True,
        {'dp': dp}
    )

    sw.train(train_dataset_file, dev_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu, not silent, seed)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu, not silent)

    print(results)


@ud.group()
def xlm_r():
    pass


@xlm_r.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, dev_dataset_file, multi_gpu):
    results = UDBERTSystemWrapper.tune(
        'xlm-roberta-base',
        None,
        False,
        train_dataset_file,
        dev_dataset_file,
        multi_gpu
    )

    print(max(results, key=lambda x: x[0]))


@xlm_r.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.argument('test_dataset_file', type=click.File('r'), default='data/ud/test.conllu')
@click.option('--batch-size', type=int, default=4)
@click.option('--lr', type=float, default=2e-05)
@click.option('--dp', type=float, default=0)
@click.option('--grad-accumulation-steps', type=int, default=4)
@click.option('--multi-gpu', is_flag=True)
@click.option('--silent', is_flag=True)
@click.option('--seed', type=int, default=0)
def run(train_dataset_file, dev_dataset_file, test_dataset_file, batch_size, lr, dp, grad_accumulation_steps,
        multi_gpu, silent, seed):
    sw = UDBERTSystemWrapper(
        'xlm-roberta-base',
        None,
        False,
        {'dp': dp}
    )

    sw.train(train_dataset_file, dev_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu, not silent, seed)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu, not silent)

    print(results)


@ud.group()
def rnn():
    pass


@rnn.command()
@click.argument('tmp_download_path', type=str, default='data')
@click.argument('embeddings_save_path', type=str, default='data/ud/ud_ft.pkl')
@click.argument('dataset_file_paths', type=str, nargs=-1)
def download_embeddings(tmp_download_path, embeddings_save_path, dataset_file_paths):
    download_model('el', tmp_download_path, if_exists='ignore')
    ft = fasttext.load_model(f'{tmp_download_path}/cc.el.300.bin')

    if not dataset_file_paths:
        dataset_file_paths = [f'data/ud/{ds}.conllu' for ds in ('train', 'dev', 'test')]

    vocab = set()
    for p in dataset_file_paths:
        with open(p) as fr:
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

    with open(embeddings_save_path, 'wb') as fw:
        pickle.dump((np.array(word_vectors), w2i, i2w), fw)


@rnn.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('char_vocab_save_path', type=str, default='data/ud/char_voc.pkl')
def create_char_vocab(train_dataset_file, char_vocab_save_path):
    vocab = set()
    for e in parse_incr(train_dataset_file):
        for t in e:
            vocab.update(list(t['form']))

    c2i = {c: i + 4 for i, c in enumerate(vocab)}
    c2i['<PAD>'] = 0
    c2i['<UNK>'] = 1
    c2i['<SOW>'] = 2
    c2i['<EOW>'] = 3

    with open(char_vocab_save_path, 'wb') as fw:
        pickle.dump(c2i, fw)


@rnn.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.argument('embeddings_file', type=click.File('rb'), default='data/ud/ud_ft.pkl')
@click.argument('char_vocab_file', type=click.File('rb'), default='data/ud/char_voc.pkl')
@click.option('--multi-gpu', is_flag=True)
def tune(train_dataset_file, dev_dataset_file, embeddings_file, char_vocab_file, multi_gpu):
    embeddings, w2i, _ = pickle.load(embeddings_file)
    c2i = pickle.load(char_vocab_file)

    results = UDRNNSystemWrapper.tune(
        embeddings,
        w2i,
        c2i,
        train_dataset_file,
        dev_dataset_file,
        multi_gpu
    )

    print(max(results, key=lambda x: x[0]))


@rnn.command()
@click.argument('train_dataset_file', type=click.File('r'), default='data/ud/train.conllu')
@click.argument('dev_dataset_file', type=click.File('r'), default='data/ud/dev.conllu')
@click.argument('test_dataset_file', type=click.File('r'), default='data/ud/test.conllu')
@click.argument('embeddings_file', type=click.File('rb'), default='data/ud/ud_ft.pkl')
@click.argument('char_vocab_file', type=click.File('rb'), default='data/ud/char_voc.pkl')
@click.option('--batch-size', type=int, default=64)
@click.option('--lr', type=float, default=1e-02)
@click.option('--dp', type=float, default=0.0)
@click.option('--rnn-hs', type=int, default=100)
@click.option('--char-emb-size', type=int, default=30)
@click.option('--grad-accumulation-steps', type=int, default=1)
@click.option('--multi-gpu', is_flag=True)
@click.option('--silent', is_flag=True)
@click.option('--seed', type=int, default=0)
def run(train_dataset_file, dev_dataset_file, test_dataset_file, embeddings_file, char_vocab_file, batch_size, lr, dp,
        rnn_hs, char_emb_size, grad_accumulation_steps, multi_gpu, silent, seed):
    embeddings, w2i, _ = pickle.load(embeddings_file)
    c2i = pickle.load(char_vocab_file)

    sw = UDRNNSystemWrapper(
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

    sw.train(train_dataset_file, dev_dataset_file, lr, batch_size, grad_accumulation_steps, multi_gpu, not silent, seed)
    results = sw.evaluate(test_dataset_file, batch_size, multi_gpu, not silent)

    print(results)


if __name__ == '__main__':
    ud()
