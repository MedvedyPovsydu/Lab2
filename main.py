from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path("Lab2_Media")
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
LOG_FILE = BASE_DIR / "logs" / "history.txt"

FILTER_NAME = "Серый мир + Canny"


def gray_world(img_bgr):
    img = img_bgr.astype(np.float32)
    mean_b, mean_g, mean_r = (img[:, :, c].mean() for c in range(3))
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    eps = 1e-6
    img[:, :, 0] *= mean_gray / max(mean_b, eps)
    img[:, :, 1] *= mean_gray / max(mean_g, eps)
    img[:, :, 2] *= mean_gray / max(mean_r, eps)

    return np.clip(img, 0, 255).astype(np.uint8)


def canny_edges(img_bgr, low=100, high=200):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    return cv2.Canny(blurred, low, high)


def write_log(filename, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] Файл: {filename} | Фильтр: {FILTER_NAME} | Статус: {status}\n"
    try:
        with open(LOG_FILE, "a", encoding="UTF-8") as log:
            log.write(line)
    except OSError:
        print("Не удалось записать лог")


def main():
    filename = input("Введите название файла: ").strip()
    path = INPUT_DIR / filename

    try:
        with open(path, "rb") as f:
            data = f.read()
    except PermissionError:
        print(f"Ошибка: Нет прав на чтение файла {filename}. Обратитесь к администратору")
        write_log(filename, "Ошибка доступа")
        return
    except FileNotFoundError:
        print("Ошибка: Файл не найден")
        write_log(filename, "Файл не найден")
        return
    except OSError as e:
        print(f"Ошибка: Не удалось открыть файл ({e.strerror})")
        write_log(filename, "Ошибка открытия")
        return

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Ошибка: Файл не является изображением")
        write_log(filename, "Неверный формат")
        return

    balanced = gray_world(img)
    edges = canny_edges(balanced)

    result_path = OUTPUT_DIR / "result.jpg"
    try:
        ok, encoded = cv2.imencode(".jpg", edges)
        if not ok:
            raise OSError("Не удалось закодировать изображение")
        with open(result_path, "wb") as out:
            out.write(encoded.tobytes())
        ok, encoded = cv2.imencode(".jpg", balanced)
        if ok:
            with open(OUTPUT_DIR / "result_balanced.jpg", "wb") as out:
                out.write(encoded.tobytes())
    except OSError as e:
        print(f"Ошибка: Не удалось сохранить результат ({e})")
        write_log(filename, "Ошибка сохранения")
        return

    print(f"Результат сохранён: {result_path}")
    write_log(filename, "Успешно")

main()
