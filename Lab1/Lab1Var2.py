import numpy as np

EPS = 1e-9

class Infeasible(Exception):
    pass

class Unbounded(Exception):
    pass

# ПРИВЕДЕНИЕ К КАНОНИЧЕСКОМУ ВИДУ
def canonical_form(A, b, signs):
    """
    Приведение системы ограничений к каноническому виду

    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    signs = list(signs)

    # Если b < 0, умножаем всё ограничение на -1
    for i in range(len(b)):
        if b[i] < 0:
            A[i] *= -1
            b[i] *= -1

            if signs[i] == "<=":
                signs[i] = ">="
            elif signs[i] == ">=":
                signs[i] = "<="

    m, n = A.shape

    names = [f"x{i + 1}" for i in range(n)]
    basis = [-1] * m
    artificial = []

    # Добавляем дополнительные переменные
    for i, sign in enumerate(signs):

        if sign in ("<=", ">="):

            column = np.zeros((m, 1))

            if sign == "<=":
                column[i] = 1
            else:
                column[i] = -1

            A = np.hstack((A, column))

            names.append(f"x{len(names) + 1}")

            # При <= новая переменная сразу базисная
            if sign == "<=":
                basis[i] = A.shape[1] - 1

    # Добавляем искусственные переменные, если в строке ещё нет базисной переменной
    for i in range(m):

        if basis[i] == -1:

            column = np.zeros((m, 1))
            column[i] = 1

            A = np.hstack((A, column))

            names.append(f"x{len(names) + 1}")

            index = A.shape[1] - 1

            basis[i] = index
            artificial.append(index)

    return A, b, basis, artificial, names

# ПОСТРОЕНИЕ СИМПЛЕКС-ТАБЛИЦЫ
def make_tableau(A, b, c, basis):
    m, n = A.shape

    T = np.zeros((m + 1, n + 1))

    T[:m, :n] = A
    T[:m, -1] = b

    # Строка целевой функции
    T[-1, :n] = -c

    # Обнуляем коэффициенты базисных переменных
    for i, j in enumerate(basis):
        T[-1] -= T[-1, j] * T[i]

    return T

# СИМПЛЕКС-МЕТОД
# Выполнение симплекс-итераций
def simplex(T, basis):

    while True:

        # Ищем входящую переменную: первый отрицательный коэффициент
        entering = None

        for j, value in enumerate(T[-1, :-1]):
            if value < -EPS:
                entering = j
                break

        # Отрицательных коэффициентов нет значит найден оптимум
        if entering is None:
            return T, basis

        # Выбираем выходящую переменную
        ratios = []

        for i in range(len(basis)):

            if T[i, entering] > EPS:

                ratio = T[i, -1] / T[i, entering]

                ratios.append((ratio, i))

        # Если положительных элементов в столбце нет, целевая функция не ограничена
        if not ratios:
            raise Unbounded("Целевая функция не ограничена.")

        _, leaving = min(ratios)

        # Разрешающий элемент
        pivot = T[leaving, entering]

        # Делим разрешающую строку
        T[leaving] /= pivot

        # Обнуляем остальные элементы разрешающего столбца
        for i in range(T.shape[0]):

            if i != leaving:
                T[i] -= T[i, entering] * T[leaving]

        # Обновляем базис
        basis[leaving] = entering

        T[np.abs(T) < EPS] = 0

# УДАЛЕНИЕ ИСКУССТВЕННЫХ ПЕРЕМЕННЫХ
def remove_artificial(T, basis, artificial, names):

    artificial = set(artificial)

    i = 0

    while i < len(basis):

        # Если искусственная переменная осталась в базисе
        if basis[i] in artificial:

            replacement = None

            # Ищем обычную переменную, которую можно ввести вместо неё
            for j in range(T.shape[1] - 1):

                if (
                    j not in artificial
                    and j not in basis
                    and abs(T[i, j]) > EPS
                ):
                    replacement = j
                    break

            if replacement is not None:

                # Делаем новый разрешающий элемент равным 1
                T[i] /= T[i, replacement]

                # Обнуляем столбец
                for r in range(T.shape[0]):

                    if r != i:
                        T[r] -= T[r, replacement] * T[i]

                basis[i] = replacement

            elif abs(T[i, -1]) < EPS:

                # Строка является избыточной
                T = np.delete(T, i, axis=0)
                basis.pop(i)

                continue

            else:

                raise Infeasible("Допустимого решения нет.")

        i += 1

    # Оставляем только неискусственные столбцы
    keep = []

    for j in range(T.shape[1] - 1):

        if j not in artificial:
            keep.append(j)

    # Соответствие старых и новых индексов
    old_to_new = {
        old: new
        for new, old in enumerate(keep)
    }

    A = T[:-1, keep]
    b = T[:-1, -1]

    basis = [
        old_to_new[j]
        for j in basis
    ]

    names = [
        names[j]
        for j in keep
    ]

    return A, b, basis, names


# ВЫВОД КАНОНИЧЕСКОГО ВИДА

def print_canonical(A, b, names):

    print("\nКанонический вид:")

    for i in range(len(b)):

        equation = ""

        for j, coef in enumerate(A[i]):

            if abs(coef) < EPS:
                continue

            value = abs(coef)

            if abs(value - 1) < EPS:
                term = names[j]
            else:
                term = f"{value:g}{names[j]}"

            if equation == "":

                if coef > 0:
                    equation = term
                else:
                    equation = "- " + term

            else:

                if coef > 0:
                    equation += " + " + term
                else:
                    equation += " - " + term

        print(f"{equation} = {b[i]:g}")


# ОСНОВНАЯ ФУНКЦИЯ РЕШЕНИЯ
def solve(c, A, signs, b, sense="min"):

# Проверка входных данных

    if sense not in ("min", "max"):
        raise ValueError("Допустимы только min и max.")

    if len(A) != len(b) or len(A) != len(signs):
        raise ValueError("Количество ограничений, знаков и ""правых частей не совпадает.")

    if any(sign not in ("<=", ">=", "=") for sign in signs):
        raise ValueError( "Допустимые знаки: <=, >=, =.")

    n = len(c)

    if any(len(row) != n for row in A):
        raise ValueError("Неверное количество коэффициентов.")

    c = np.array(c, dtype=float)

# Канонический вид

    A, b, basis, artificial, names = canonical_form(A, b, signs)

    print_canonical(A, b, names)

    print(
        "\nНачальный базис:",
        ", ".join(names[j] for j in basis)
    )

    if artificial:

        print(
            "Искусственные переменные:",
            ", ".join(names[j] for j in artificial)
        )

    else:

        print("Искусственные переменные не требуются.")

    # ШАГ I

    if artificial:

        c1 = np.zeros(A.shape[1])

        for j in artificial:
            c1[j] = -1

        T = make_tableau(A, b, c1, basis)

        T, basis = simplex(T, basis)

        # Получаем минимум вспомогательной функции
        W = -T[-1, -1]

        if abs(W) < EPS:
            W = 0

        print(f"\nW_min = {W:g}")

        # Если W > 0, исходная система несовместна
        if W > EPS:

            raise Infeasible(
                "Допустимого решения нет."
            )

        print("Допустимое решение существует.")

        # Удаляем искусственные переменные
        A, b, basis, names = remove_artificial(T, basis, artificial, names)

    # ШАГ II

    c_full = np.zeros(len(names))
    c_full[:n] = c

    if sense == "min":
        c_simplex = -c_full
    else:
        c_simplex = c_full.copy()

    T = make_tableau(A, b, c_simplex, basis)

    T, basis = simplex(T,basis)

    # ВОССТАНОВЛЕНИЕ РЕШЕНИЯ

    x_all = np.zeros(len(names))

    for i, j in enumerate(basis):

        x_all[j] = T[i, -1]

    x = x_all[:n]
    x[np.abs(x) < EPS] = 0
    Z = c @ x

    if abs(Z) < EPS:
        Z = 0


# Вывод результата
    print("\nОптимальное решение:")

    for i, value in enumerate(x):

        print(f"x{i + 1} = {value:g}")

    print(f"Z_{sense} = {Z:g}")

    return x, Z

# ИСХОДНЫЕ ДАННЫЕ


c = [1, 3, 2, 1]

A = [
    [1, 1, 0, 2],
    [0, 1, 1, 1],
    [2, 0, 1, 0]
]

signs = ["<=", "=", ">="]

b = [8, 6, 2]

sense = "min"

try:
    solve(
        c=c,
        A=A,
        signs=signs,
        b=b,
        sense=sense
    )

except Infeasible as error:
    print("\nОшибка:", error)

except Unbounded as error:
    print("\nОшибка:", error)

except ValueError as error:
    print("\nОшибка входных данных:", error)
