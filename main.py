import sys
import os.path
import numpy as np
from my_robust_hash import my_robust_hash


def get_ascii_symbols() -> list[str]:
    return [chr(i) for i in range(32, 128)]


class Memorize:
    def __init__(self, func):
        self.__func = func
        self.__cache = {}

    def __call__(self, arg):
        if arg not in self.__cache:
            self.__cache[arg] = self.__func(arg)
        return self.__cache[arg]


@Memorize
def calculate_robust_hash_for_image(file: str) -> np.array:
    hash_dim = 48
    h = my_robust_hash(file, hash_dim=hash_dim)
    return h


def sender(key_generator: int, recourse: str, message: str):
    np.random.seed(key_generator)

    files = [os.path.join(recourse, name) for name in os.listdir(recourse)
             if os.path.isfile(os.path.join(recourse, name))]

    unique_hashes = []
    for file in files:
        h = calculate_robust_hash_for_image(file)

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
    # recourse = sys.argv[1]
    recourse = './images'
    # sender(key_generator, recourse, message)
    # receiver()


if __name__ == '__main__':
    main()
