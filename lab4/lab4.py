import sys

import numpy as np
import sympy as sp
import time
import matplotlib.pyplot as plt

import colors as c

from typing import Iterable, Collection, Callable, Any
from types import ModuleType

# промежуток для рандомных чисел векторов сдвига
SHIFT_MAX: int = 5
SHIFT_MIN: int = -SHIFT_MAX
SEED = None

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(seed=SEED)

# константа как в UE5
SMALL_NUMBER = 1e-8

# ue_is_nearly_zero(val: float) -> bool -- копирка bool IsNearlyZero(float, float) из UE5
def is_zero(val: float, small_num: float = SMALL_NUMBER) -> bool: return abs(val) <= small_num
def is_equal(a: float, b: float, small_num: float = SMALL_NUMBER) -> bool: return abs(a - b) <= small_num
def is_pi(val: float, small_num: float = SMALL_NUMBER) -> bool: return abs(val - np.pi) <= small_num

RED = c.style(c.red)
GREEN = c.style(c.green)
YELLOW = c.style(c.yellow)
BOLD = c.style(c.bold)
CLS = c.style(c.clear)

def red(val) -> str: return f'{RED}{val}{CLS}'
def green(val) -> str: return f'{GREEN}{val}{CLS}'
def yellow(val) -> str: return f'{YELLOW}{val}{CLS}'
def bold(val) -> str: return f'{BOLD}{val}{CLS}'

def check(expr: bool) -> str:
	return f'{c.style(c.bold, c.green)}OK{CLS}' if expr else f'{c.style(c.bold, c.red)}FAIL{CLS}'

K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])

def rodrigues(w):
    """Формула Родрига из лекции 3 — понадобится для задания 4."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


#region TASK 1


# src - массив исходных точек. (x, y)
# dst - массив целевых точек. (u, v)
def make_A(src: Iterable, dst: Iterable) -> np.ndarray:
	A = []
	for (x, y), (u, v) in zip(src, dst):
		# коэфициенты системы линейных уравнений  
		# img1.png
		A += [[-x, -y, -1, 0, 0, 0, u*x, u*y, u]]
		A +=[[0, 0, 0, -x, -y, -1, v*x, v*y, v]]
	return np.array(A, dtype=float)

def dlt_homography(src, dst): 
	A = make_A(src, dst)
	# решение системы уравнений и преобразование в матрицу 3 на 3
	# берется последняя строка,
	# именно там длина вектора -> 1
	# а определитель -> 0
	H: np.ndarray = (np.linalg.svd(A)[2][-1])
	# print('len H = %f' % np.linalg.norm(H)) # len H = 1.000000
	
	H = H.reshape(3,3)	
	# print('det H = %f' % np.linalg.det(H)) # det H = 0.000005
	return H / H[2, 2]

def apply_H(H, pts):
	h = np.hstack([np.atleast_2d(pts), np.ones((len(np.atleast_2d(pts)), 1))])
	q = h @ H.T
	return q[:, :2] / q[:, 2:3]


def project_plane(M, R, t):
    X = np.hstack([M, np.zeros((len(M), 1))])
    Xc = X @ R.T + t
    uvw = Xc @ K.T
    return uvw[:, :2] / uvw[:, 2:3]

def task_1():
	# точки квадрата со сторонами 0.2 и центром в (0,0)
	# позаимствовал из решения
	M = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])

	# просто случайные 3 числа
	# w_true = rng.uniform(-0.5, 0.5, size=3)
	w_true = np.array([0.35, -0.55, 0.25])
	R_true = rodrigues(w_true)
	t_true = np.array([0.02, -0.01, 0.9])

	img = project_plane(M, R_true, t_true)
	H = dlt_homography(M, img)
	print(f'{YELLOW}Матрица гомографии{CLS}')
	print(H)

	err_1 = np.abs(apply_H(H, M) - img).max()
	print(f'\n{YELLOW}Проверка невязки{CLS}')
	print(f'на 4 опорных точках\t= {err_1:.3e}\t=> {check(is_zero(err_1, 9e-10))}')

	M_many = rng.uniform(-.1, .1, (50,2))
	err_2 = np.abs(apply_H(H, M_many) - project_plane(M_many, R_true, t_true)).max()
	print(f'на 50 точках маркера\t= {err_2:.3e}\t=> {check(is_zero(err_2, 9e-10))}')


#endregion


def task_2(): pass
def task_3(): pass
def task_4(): pass
def task_5(): pass
def task_6(): pass
def task_7(): pass

LAB_TASKS = [task_1, task_2, task_3, task_4, task_5, task_6, task_7]

# int main(int argc, char** argv) ;P
if __name__ == '__main__':
	if len(sys.argv) == 1:
		for i in range(len(LAB_TASKS)):
			print('\x1b[1;33mЗадание %d\x1b[0m' % (i+1))
			LAB_TASKS[i]()
			print('\n')

	if len(sys.argv) == 2:
		id = int(sys.argv[1])
		print('\x1b[1;33mЗадание %d\x1b[0m' % id)
		LAB_TASKS[id - 1]()

#endregion