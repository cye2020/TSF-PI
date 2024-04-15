from itertools import accumulate


def change2close(last_close, changes):
    return list(accumulate([last_close] + changes, lambda x, y: round(x * (1 + y / 100), 2)))[1:]