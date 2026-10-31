"""
Модуль my_sort: реализация Smoothsort с визуализацией, файловым вводом/выводом.

Предоставляет функцию smoothsort с поддержкой:
- сортировки по возрастанию/убыванию,
- пользовательских key/cmp,
- пошаговой визуализации (обмены, состояние массива),
- чтения/записи данных из/в файл.
"""

from typing import List, Optional, Callable, Any, Tuple


def _cmp_to_key(cmp_func: Callable[[Any, Any], int]) -> Callable:
    """
    Преобразует функцию сравнения (cmp) в функцию-ключ для сортировки.

    Функция возвращает класс-обёртку, реализующую сравнение через cmp_func.
    Позволяет использовать стиль сравнения из Python2 в ф-ях сор-ки Python3.
    """

    class K:
        __slots__ = ["obj"]

        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def __lt__(self, other: "K") -> bool:
            return cmp_func(self.obj, other.obj) < 0

    return K


def precompute_leo(n: int) -> List[int]:
    """
    Вычисляет числа Леонардо до заданного предела.

    Числа Леонардо определяются рекуррентно:
        L(0) = 1
        L(1) = 1
        L(k) = L(k-1) + L(k-2) + 1

    Возвращает список [L(0), L(1), ..., L(m)], где L(m) <= n.
    """
    if n < 1:
        return []

    leo = [1, 1]
    while True:
        nxt = leo[-1] + leo[-2] + 1
        if nxt > n:
            break
        leo.append(nxt)
    return leo


class StepVisualizer:
    """
    Утилита для пошаговой визуализации процесса сортировки.

    Позволяет выводить состояние массива и фиксировать каждый обмен элементов.
    Используется как коллбэк для smoothsort.
    """

    def __init__(self) -> None:
        """Инициализирует визуализатор, устанавливая счётчик шагов в ноль."""
        self.step_count = 0

    def on_swap(self, i: int, j: int) -> None:
        """
        Вызывается при каждом обмене элементов массива.

        Аргументы:
            i: индекс первого элемента.
            j: индекс второго элемента.
        """
        self.step_count += 1
        print(f"Шаг {self.step_count}: обмен индексов {i} <-> {j}")

    def show_array(self, arr: List[Any], note: str = "") -> None:
        """
        Выводит текущее состояние массива с пояснением.

        Аргументы:
            arr: список элементов.
            note: необязательная заметка (например, тип шага).
        """
        arr_vals = [str(x) for x in arr]
        prefix = f"{note}: " if note else ""
        print(f"{prefix}Массив: [{', '.join(arr_vals)}]")


def _default_key(x: Any) -> Any:
    """Функция-ключ по умолчанию: возвращает сам элемент."""
    return x


def smoothsort(
    arr: List[Any],
    reverse: bool = False,
    key: Optional[Callable] = None,
    cmp: Optional[Callable[[Any, Any], int]] = None,
    on_swap: Optional[Callable[[int, int], None]] = None,
    on_array_state: Optional[Callable[[List[Any], str], None]] = None,
) -> List[Any]:
    """
    Реализация алгоритма плавной сортировки (Smoothsort).

    Smoothsort — адаптивный алгоритм сортировки на основе куч Леонардо.
    Эффективно сортирует почти упорядоченные массивы за O(n log n) в худш. сл.
    и близок к O(n) на почти отсортированных данных.

    Аргументы:
        arr: список элементов для сортировки.
        reverse: если True, сортировка по убыванию.
        key: функция извлечения ключа сравнения.
        cmp: функция сравнения (возвращает -1, 0, 1).
        on_swap: коллбэк, вызываемый при каждом обмене индексов.
        on_array_state: коллбэк, вызываемый для вывода состояния массива.

    Возвращает:
        Новый отсортированный список.
    """
    # Выбор функции ключа
    if cmp is not None:
        key_func = _cmp_to_key(cmp)
    elif key is not None:
        key_func = key
    else:
        key_func = _default_key

    # Упаковываем элементы в кортежи (ключ, исходный_индекс, объект)
    a: List[Tuple[Any, int, Any]] = [
        (key_func(x), i, x) for i, x in enumerate(arr)
    ]
    n = len(a)
    if n <= 1:
        return [x[2] for x in a]

    leo = precompute_leo(n)

    def sift_down(root: int, k: int) -> None:
        """
        Просеивание элемента вниз внутри кучи Леонардо.

        Аргументы:
            root: индекс корня текущей кучи.
            k: порядок кучи (соответствует числу Леонардо).
        """
        while k > 1:
            left_size = leo[k - 1]
            # right_size не используется напрямую
            left_idx = root + 1
            right_idx = root + left_size + 1

            largest = root
            if a[left_idx][0] > a[largest][0]:
                largest = left_idx
            if right_idx < root + leo[k] and a[right_idx][0] > a[largest][0]:
                largest = right_idx

            if largest == root:
                break

            a[root], a[largest] = a[largest], a[root]
            if on_swap is not None:
                on_swap(root, largest)
            if on_array_state is not None:
                vals = [x[2] for x in a]
                on_array_state(
                    vals,
                    f"просеивание вниз: {root}<->{largest}",
                )

            if largest == left_idx:
                k -= 1
                root = left_idx
            else:
                k -= 2
                root = right_idx

    def trinkle(
        root: int,
        k: int,
        head: int,
        step_sizes: List[int],
        step_heads: List[int],
    ) -> None:
        """
        Восстанавливает порядок корней соседних куч (просеивание вверх).

        Аргументы:
            root: индекс текущего корня.
            k: порядок текущей кучи.
            head: индекс «головного» корня для сравнения.
            step_sizes: список порядков куч.
            step_heads: список индексов корней куч.
        """
        while k > 0 and a[root][0] <= a[head][0]:
            if on_swap is not None:
                on_swap(root, head)
            a[root], a[head] = a[head], a[root]
            if on_array_state is not None:
                vals = [x[2] for x in a]
                on_array_state(
                    vals,
                    f"просеивание вверх (trinkle): {root}<->{head}",
                )
            sift_down(root, k)
            head = root
            k -= 1
            if k > 0:
                root = step_heads[-k - 1]

    sizes: List[int] = []
    heads: List[int] = []

    # Фаза 1: построение леса куч
    for i in range(n):
        sizes.append(0)
        heads.append(i)
        sift_down(heads[-1], sizes[-1])

        j = len(sizes) - 1
        while j > 0 and a[heads[j]][0] <= a[heads[j - 1]][0]:
            a[heads[j]], a[heads[j - 1]] = a[heads[j - 1]], a[heads[j]]
            if on_swap is not None:
                on_swap(heads[j], heads[j - 1])
            if on_array_state is not None:
                vals = [x[2] for x in a]
                on_array_state(
                    vals,
                    f"корни куч (trinkle): {heads[j]}<->{heads[j - 1]}",
                )
            sift_down(heads[j - 1], sizes[j - 1])
            j -= 1

        # Слияние куч при совпадении порядков
        while (
            len(sizes) >= 3
            and sizes[-1] == sizes[-2] == sizes[-3] - 1
        ):
            k1 = sizes[-3]
            h1 = heads[-3]

            sizes[-3] = k1 + 1
            heads[-3] = h1
            del sizes[-2:]
            del heads[-2:]

            sift_down(h1, sizes[-1])

            j = len(sizes) - 1
            while j > 0 and a[heads[j]][0] <= a[heads[j - 1]][0]:
                a[heads[j]], a[heads[j - 1]] = a[heads[j - 1]], a[heads[j]]
                if on_swap is not None:
                    on_swap(heads[j], heads[j - 1])
                if on_array_state is not None:
                    vals = [x[2] for x in a]
                    on_array_state(
                        vals,
                        f"корни после слияния: {heads[j]}<->{heads[j - 1]}",
                    )
                sift_down(heads[j - 1], sizes[j - 1])
                j -= 1

    # Фаза 2: извлечение максимумов
    while sizes:
        k_last = sizes[-1]
        root_last = heads[-1]
        end = n - len(sizes)

        a[root_last], a[end] = a[end], a[root_last]
        if on_swap is not None:
            on_swap(root_last, end)
        if on_array_state is not None:
            vals = [x[2] for x in a]
            on_array_state(
                vals,
                f"извлечение максимума: {root_last}<->{end}",
            )

        sizes.pop()
        heads.pop()

        if k_last > 1:
            right_root = root_last + leo[k_last - 1]
            sift_down(right_root, k_last - 2)
            sift_down(root_last, k_last - 1)

            if a[root_last][0] < a[right_root][0]:
                a[root_last], a[right_root] = (
                    a[right_root],
                    a[root_last],
                )
                if on_swap is not None:
                    on_swap(root_last, right_root)
                if on_array_state is not None:
                    vals = [x[2] for x in a]
                    on_array_state(
                        vals,
                        f"обмен корней: {root_last}<->{right_root}",
                    )
                sift_down(right_root, k_last - 2)

    result = [x[2] for x in a]
    if reverse:
        result.reverse()
    return result


def my_sort(
    array: List[Any],
    reverse: bool = False,
    key: Optional[Callable] = None,
    cmp: Optional[Callable[[Any, Any], int]] = None,
) -> List[Any]:
    """
    Обёртка для вызова плавной сортировки.

    Аргументы:
        array: список элементов для сортировки.
        reverse: сортировать по убыванию, если True.
        key: функция извлечения ключа.
        cmp: функция сравнения.

    Возвращает:
        Отсортированный список.
    """
    return smoothsort(array, reverse=reverse, key=key, cmp=cmp)


def read_numbers_from_file(path: str) -> List[Any]:
    """Читает данные из файла, пытаясь определить тип (int, float или str)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        tokens = content.split()
        try:
            return [int(x) for x in tokens]
        except ValueError:
            try:
                return [float(x) for x in tokens]
            except ValueError:
                return tokens


def write_data_to_file(path: str, data: List[Any]) -> None:
    """Записывает список данных в файл через пробел."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(str(x) for x in data))


if __name__ == "__main__":
    print("Выберите режим:")
    print("1 — ввод с клавиатуры + подробная визуализация (массив + обмены)")
    print("2 — чтение из input.txt, запись в output.txt (без визуализации)")
    mode = input("Режим (1 или 2): ").strip()

    data: List[Any] = []

    if mode == "1":
        raw = input("Введите элементы через пробел: ").strip()
        if not raw:
            print("Ничего не введено.")
            exit(0)

        tokens = raw.split()
        try:
            data = [int(x) for x in tokens]
            print("Определён тип: целые числа.")
        except ValueError:
            try:
                data = [float(x) for x in tokens]
                print("Определён тип: вещественные числа.")
            except ValueError:
                data = tokens
                print("Определён тип: текст.")

        order = input("Порядок (1=по возрастанию, 2=по убыванию): ").strip()
        reverse = order == "2"

        visualizer = StepVisualizer()

        sorted_data = smoothsort(
            data,
            reverse=reverse,
            on_swap=visualizer.on_swap,
            on_array_state=visualizer.show_array,
        )

        print("\nОтсортированный массив:", sorted_data)
        print(f"Всего обменов: {visualizer.step_count}")

    elif mode == "2":
        input_path = "input.txt"
        output_path = "output.txt"

        try:
            data = read_numbers_from_file(input_path)
            if not data:
                print("Файл input.txt пуст.")
                exit(0)
            print(f"Данные прочитаны ({len(data)} элементов).")
        except FileNotFoundError:
            print(
                f"Файл {input_path} не найден. Создайте его, запишите данные."
            )
            exit(1)

        order = input("Порядок (1=по возрастанию, 2=по убыванию): ").strip()
        reverse = order == "2"

        sorted_data = my_sort(data, reverse=reverse)
        write_data_to_file(output_path, sorted_data)
        print(f"Готово: результат записан в {output_path}")
    else:
        print("Неверный режим.")
