def pad_to_max(lst, max_len=None, pad_value=0):
    pad = len(max(lst, key=len))
    if max_len is not None:
        pad = min(max_len, pad)

    return [i + [pad_value for _ in range(pad - len(i))] if len(i) <= pad else i[:pad] for i in lst]