import sys
import os.path
import numpy as np
from my_robust_hash import my_robust_hash


def get_ascii_symbols() -> list[str]:
    return [chr(i) for i in range(32, 128)]


def sender(key_generator: int):
    np.random.seed(key_generator)

    # directory = sys.argv[1]
    directory = './images'
    files = [os.path.join(directory, name) for name in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, name))]

    unique_hashes = []
    for file in files:
        h = my_robust_hash(file, hash_dim=48)
        unique_hashes.extend(h)
    unique_hashes = list(set(unique_hashes))

    ascii_symbols = get_ascii_symbols()

    if len(unique_hashes) < len(ascii_symbols):
        raise ValueError()

    np.random.shuffle(unique_hashes)
    d = dict(zip(ascii_symbols, unique_hashes[:len(ascii_symbols)]))

    np.random.seed()


def receiver():
    pass


def main():
    key_generator = 215351
    sender(key_generator)
    # receiver()


if __name__ == '__main__':
    main()
