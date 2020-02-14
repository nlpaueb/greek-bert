import glob
import random

import sentencepiece as spm

MODEL_PREFIX = "sentencepiece_el"
VOC_SIZE = 35000
SENTENCES_SIZE = 3000000

filenames = glob.glob('/home/chalkidis/greek_corpora/*/*')

SPM_COMMAND = ('--input={} --model_prefix={} '
               '--input_sentence_size={} '
               '--vocab_size={} '
               '--shuffle_input_sentence=true').format(','.join(filenames), MODEL_PREFIX, SENTENCES_SIZE, VOC_SIZE)

spm.SentencePieceTrainer.Train(SPM_COMMAND)


def read_sentencepiece_vocab(filepath):
    voc = []
    with open(filepath, encoding='utf-8') as fi:
        for line in fi:
            voc.append(line.split("\t")[0])
    return voc


snt_vocab = read_sentencepiece_vocab("{}.vocab".format(MODEL_PREFIX))
print("Learnt vocab size: {}".format(len(snt_vocab)))
print("Sample tokens: {}".format(random.sample(snt_vocab, 10)))


def parse_sentencepiece_token(token):
    if token.startswith("▁"):
        return token[1:]
    else:
        return "##" + token


bert_vocab = list(map(parse_sentencepiece_token, snt_vocab))
# ctrl_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# bert_vocab = ctrl_symbols + bert_vocab
VOC_FNAME = "vocab_el.txt"

with open(VOC_FNAME, "w") as fo:
    for token in bert_vocab:
        fo.write(token + "\n")
