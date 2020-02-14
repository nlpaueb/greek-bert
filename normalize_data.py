import glob
from multiprocessing import Pool
import unicodedata

filenames = glob.glob('/home/chalkidis/greek_corpora/common_crawl_shards/*')
filenames += glob.glob('/home/chalkidis/greek_corpora/europarl_shards/*')
filenames += glob.glob('/home/chalkidis/greek_corpora/wikipedia_shards/*')

def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


def _run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]

def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()

def normalize(filename):
    output_file = open(filename.replace('greek_corpora', 'greek_corpora_norm'), 'w', encoding='utf8')
    with open(filename, encoding='utf8') as file:
        for line in file.readlines():
            tokens = line.lower().split()
            splited_tokens = []
            for token in tokens:
                splited_tokens.extend(_run_split_on_punc(token))
            line = ' '.join(splited_tokens)
            line = strip_accents_and_lowercase(line)
            if line.endswith('\n'):
                output_file.write(line)
            else:
                output_file.write(line+'\n')
    output_file.close()


with Pool(processes=10) as pool:
    pool.map(normalize, filenames)
