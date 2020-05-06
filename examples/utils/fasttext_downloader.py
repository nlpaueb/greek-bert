# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in https://github.com/facebookresearch/fastText/LICENSE

import sys
import shutil
import os
import gzip

from urllib.request import urlopen

valid_lang_ids = {"af", "sq", "als", "am", "ar", "an", "hy", "as", "ast",
                  "az", "ba", "eu", "bar", "be", "bn", "bh", "bpy", "bs",
                  "br", "bg", "my", "ca", "ceb", "bcl", "ce", "zh", "cv",
                  "co", "hr", "cs", "da", "dv", "nl", "pa", "arz", "eml",
                  "en", "myv", "eo", "et", "hif", "fi", "fr", "gl", "ka",
                  "de", "gom", "el", "gu", "ht", "he", "mrj", "hi", "hu",
                  "is", "io", "ilo", "id", "ia", "ga", "it", "ja", "jv",
                  "kn", "pam", "kk", "km", "ky", "ko", "ku", "ckb", "la",
                  "lv", "li", "lt", "lmo", "nds", "lb", "mk", "mai", "mg",
                  "ms", "ml", "mt", "gv", "mr", "mzn", "mhr", "min", "xmf",
                  "mwl", "mn", "nah", "nap", "ne", "new", "frr", "nso",
                  "no", "nn", "oc", "or", "os", "pfl", "ps", "fa", "pms",
                  "pl", "pt", "qu", "ro", "rm", "ru", "sah", "sa", "sc",
                  "sco", "gd", "sr", "sh", "scn", "sd", "si", "sk", "sl",
                  "so", "azb", "es", "su", "sw", "sv", "tl", "tg", "ta",
                  "tt", "te", "th", "bo", "tr", "tk", "uk", "hsb", "ur",
                  "ug", "uz", "vec", "vi", "vo", "wa", "war", "cy", "vls",
                  "fy", "pnb", "yi", "yo", "diq", "zea"}


def _print_progress(downloaded_bytes, total_size):
    percent = float(downloaded_bytes) / total_size
    bar_size = 50
    bar = int(percent * bar_size)
    percent = round(percent * 100, 2)
    sys.stdout.write(" (%0.2f%%) [" % percent)
    sys.stdout.write("=" * bar)
    sys.stdout.write(">")
    sys.stdout.write(" " * (bar_size - bar))
    sys.stdout.write("]\r")
    sys.stdout.flush()

    if downloaded_bytes >= total_size:
        sys.stdout.write('\n')


def _download_file(url, out_folder, write_file_name, chunk_size=2 ** 13):
    print("Downloading %s" % url)

    response = urlopen(url)

    if hasattr(response, 'getheader'):
        file_size = int(response.getheader('Content-Length').strip())
    else:
        file_size = int(response.info().getheader('Content-Length').strip())
    downloaded = 0
    download_file_name = write_file_name + ".part"
    with open(f'{out_folder}/{download_file_name}', 'wb') as f:
        while True:
            chunk = response.read(chunk_size)
            downloaded += len(chunk)
            if not chunk:
                break
            f.write(chunk)
            _print_progress(downloaded, file_size)

    os.rename(f'{out_folder}/{download_file_name}', f'{out_folder}/{write_file_name}')


def _download_gz_model(out_folder, gz_file_name, if_exists):
    if os.path.isfile(f'{out_folder}/{gz_file_name}'):
        if if_exists == 'ignore':
            return True
        elif if_exists == 'strict':
            print("gzip File exists. Use --overwrite to download anyway.")
            return False
        elif if_exists == 'overwrite':
            pass

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/%s" % gz_file_name
    _download_file(url, out_folder, gz_file_name)

    return True


def download_model(lang_id, out_folder='.', if_exists='strict'):
    """
        Download pre-trained common-crawl vectors from fastText's website
        https://fasttext.cc/docs/en/crawl-vectors.html
    """
    if lang_id not in valid_lang_ids:
        raise Exception("Invalid lang id. Please select among %s" %
                        repr(valid_lang_ids))

    os.makedirs(out_folder, exist_ok=True)

    file_name = f"cc.{lang_id}.300.bin"
    gz_file_name = f"{file_name}.gz"

    if os.path.isfile(f'{out_folder}/{file_name}'):
        if if_exists == 'ignore':
            return file_name
        elif if_exists == 'strict':
            print("File exists. Use --overwrite to download anyway.")
            return
        elif if_exists == 'overwrite':
            pass

    if _download_gz_model(out_folder, gz_file_name, if_exists):
        with gzip.open(f'{out_folder}/{gz_file_name}', 'rb') as f:
            with open(f'{out_folder}/{file_name}', 'wb') as f_out:
                shutil.copyfileobj(f, f_out)

    return file_name
