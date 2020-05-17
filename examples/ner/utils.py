def parse_ner_dataset_file(f):
    tokens = []
    for i, l in enumerate(f):
        l_split = l.split()
        if len(l_split) == 0:
            yield tokens
            tokens.clear()
            continue
        if len(l_split) != 2:
            continue  # todo: fix this
        else:
            tokens.append({'text': l_split[0], 'label': l_split[1]})
    if tokens:
        yield tokens
