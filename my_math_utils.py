import numpy as np


def my_dct(grayMat: np.uint8) -> np.array:
    """ Двумерное дискретное косинусное преобразование """
    assert grayMat.shape == (8, 8)
    assert grayMat.dtype == np.uint8
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


def my_PCA(vectors: np.array, k: int) -> np.array:
    # Вычисляем среднее по векторам
    mean = np.mean(vectors, axis=0)
    # Центрирование данных (sum(Xi)=0)
    centered_data = vectors - mean
    # Вычисление ковариационной матрицы
    covariance_matrix = np.cov(centered_data, rowvar=False)
    # Вычисление собственных значений и собственных векторов ковариационной матрицы
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    del covariance_matrix
    # Сортировка собственных векторов в порядке убывания собственных значений
    indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues, sorted_eigenvectors = eigenvalues[indices], eigenvectors[:, indices]

    # Выбор первых k собственных векторов
    selected_eigenvectors = sorted_eigenvectors[:, :k]

    # Преобразование данных в новое пространство признаков
    X_reduced = np.dot(centered_data, selected_eigenvectors)
    # Обратное преобразование данных в исходное пространство признаков
    X_recovered = np.dot(X_reduced, selected_eigenvectors.T) + mean
    return X_recovered
