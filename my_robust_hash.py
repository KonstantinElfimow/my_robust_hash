import numpy as np
import cv2
from skimage import io
from my_math_utils import my_PCA, my_dct


def compare_images(old_image: np.array, new_image: np.array) -> None:
    reconstructed_image = np.zeros(old_image.shape, dtype=np.uint8)
    count = 0
    for i in range(8, old_image.shape[0], 8):
        for j in range(8, old_image.shape[1], 8):
            reconstructed_image[i - 8: i, j - 8: j] = new_image[count, :, :]
            count += 1
    io.imshow(np.hstack((old_image, reconstructed_image)))
    io.show()


def my_pHash(block: np.array) -> np.uint64:
    """ Перцептивное хэширование """
    assert block.shape == (8, 8)
    assert block.dtype == np.uint8

    dct = my_dct(block)
    dct = dct.reshape(-1)
    x = (np.arange(64))[dct >= dct.mean()]
    h = np.uint64(np.sum(np.power(2, x)))
    return h


def my_robust_hash(file: str, *, hash_dim: int = 1) -> np.array:
    # Загрузка изображения
    img = cv2.imread(file)
    # Преобразуем в оттенки серого
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Разбиение на блоки размером 8x8
    blocks = []
    for i in range(8, img.shape[0], 8):
        for j in range(8, img.shape[1], 8):
            block = img[i - 8: i, j - 8: j]
            blocks.append(block)
    vectors = np.array(blocks, dtype=np.uint8).reshape(-1, 64)
    del blocks

    # Восстановленные блоки 8x8
    reconstructed = np.array(my_PCA(vectors, vectors.shape[1] // 4), dtype=np.uint8).reshape(-1, 8, 8)

    # Оценка информативности восстановленных блоков по дисперсии
    block_scores = np.array([np.var(block, axis=(0, 1)) for block in reconstructed])

    # Выбор наиболее информативных блоков
    best_block_indices = (block_scores.argsort()[::-1])[:hash_dim]
    best_blocks = reconstructed[best_block_indices].copy()

    # Вычисление pHash для самых информативных восстановленных блоков
    hash_values = np.array([my_pHash(block) for block in best_blocks])
    # Было / стало
    # compare_images(img, reconstructed)

    return hash_values
