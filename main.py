import numpy as np
import cv2
import sys
import os.path
import skimage.io as skio


def my_dct(grayMat: np.uint8):
    """ Двумерное дискретное косинусное преобразование """
    assert grayMat.shape == (8, 8)
    # Создаем матрицу коэффициентов
    coeffs = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == 0:
                coeffs[i, j] = 1 / np.sqrt(8)
            else:
                coeffs[i, j] = np.sqrt(2 / 8) * np.cos((np.pi * (2 * j + 1) * i) / (2 * 8))

    # Вычисляем DCT для блока
    dct_block = np.dot(np.dot(coeffs, grayMat), coeffs.T)

    return dct_block


def my_hamming(h_1: np.uint64, h_2: np.uint64) -> np.uint8:
    d: np.uint8 = np.uint8(0)
    h: np.uint64 = np.bitwise_xor(h_1, h_2)
    while h:
        h = np.bitwise_and(np.uint64(h), np.uint64(h - 1))
        d += 1
    return d


def my_PCA(data: np.array, k: int) -> np.array:
    # Центрирование данных
    centered_data = data - np.mean(data, axis=0)
    print(centered_data.shape)
    # Вычисление ковариационной матрицы
    covariance_matrix = np.cov(centered_data, rowvar=False)
    print(centered_data.shape)
    # Вычисление собственных значений и собственных векторов ковариационной матрицы
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Сортировка собственных векторов в порядке убывания собственных значений
    indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues, sorted_eigenvectors = eigenvalues[indices], eigenvectors[:, indices]

    # Выбор первых k собственных векторов
    selected_eigenvectors = sorted_eigenvectors[:, :k]

    # Преобразование данных в новое пространство признаков
    X_reduced = np.dot(centered_data, selected_eigenvectors)
    # Обратное преобразование данных в исходное пространство признаков
    X_recovered = np.dot(X_reduced, selected_eigenvectors.T) + np.mean(data, axis=0)
    return X_recovered


def my_robust_hash(file: str) -> np.array:
    # Загрузка изображения
    img = cv2.imread(file)
    img = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2GRAY)
    # Разбиение на блоки размером 8x8
    blocks = []
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            block = img[i: i + 8, j: j + 8]
            blocks.append(block)
    vectors = np.array(blocks, dtype=np.uint8).reshape(-1, 64)

    reconstructed = np.uint8(my_PCA(vectors, 3))
    # print('Новое', reconstructed[:])
    # print('Старое', vectors[:])
    significant_blocks = np.array(reconstructed.reshape(-1, 8, 8), dtype=np.uint8)

    # Вычисление pHash
    hash_values = []
    for block in significant_blocks:
        dct = my_dct(block)
        dct = dct.reshape(-1)
        # print(dct, dct.mean())

        x = (np.arange(64))[dct >= dct.mean()]
        h = np.uint64(np.sum(np.power(2, x)))
        hash_values.append(h)

    return np.array(hash_values, dtype=np.uint64)


def main():
    # directory = sys.argv[1]
    directory = './images'
    files = [os.path.join(directory, name) for name in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, name))]
    hashes = [my_robust_hash(image) for image in files]
    print(hashes)


if __name__ == '__main__':
    main()
