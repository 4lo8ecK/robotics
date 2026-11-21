import sys

import numpy as np
import sympy as sp
import time
import matplotlib.pyplot as plt

import colors as c

from typing import Iterable, Collection, Callable, Any
from types import ModuleType

arr_t = np.ndarray

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

def rodrigues(w: arr_t) -> arr_t:
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
def make_A(src: arr_t, dst: arr_t) -> arr_t:
	A = []
	for (x, y), (u, v) in zip(src, dst):
		# коэфициенты системы линейных уравнений  
		# img1.png
		A += [[-x, -y, -1, 0, 0, 0, u*x, u*y, u]]
		A +=[[0, 0, 0, -x, -y, -1, v*x, v*y, v]]
	return np.array(A, dtype=float)

def dlt_homography(src: arr_t, dst: arr_t) -> arr_t: 
	A = make_A(src, dst)
	# решение системы уравнений и преобразование в матрицу 3 на 3
	# берется последняя строка,
	# именно там длина вектора -> 1
	# а определитель -> 0
	H: arr_t = (np.linalg.svd(A)[2][-1])
	# print('len H = %f' % np.linalg.norm(H)) # len H = 1.000000
	
	H = H.reshape(3,3)	
	# print('det H = %f' % np.linalg.det(H)) # det H = 0.000005
	return H / H[2, 2]

def apply_H(H: arr_t, pts: arr_t) -> arr_t:
	h = np.hstack([np.atleast_2d(pts), np.ones((len(np.atleast_2d(pts)), 1))])
	# h = np.column_stack([pts, np.ones(len(pts))])
	q = h @ H.T
	return q[:, :2] / q[:, 2:3]

def project_plane(M: arr_t, R: arr_t, t: arr_t) -> arr_t:
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

#region TASK 2

# нормализация точек методом Хартли
def normalize_pts(pts: arr_t) -> tuple[Any, Any]:
	centroid = pts.mean(axis=0)
	mean_dist = np.linalg.norm(pts - centroid, axis=1).mean()
	s = np.sqrt(2.0) / mean_dist
	# получено из формулы - img3.png
	T = np.array([
		[s, 0, -s * centroid[0]],
		[0, s, -s * centroid[1]],
		[0, 0, 1]
	])
	return (pts - centroid) * s, T

def dlt_normalized(src: arr_t, dst: arr_t) -> arr_t:
	src_n, T_src = normalize_pts(src)
	dst_n, T_dst = normalize_pts(dst)
	H_n = dlt_homography(src_n, dst_n)
	H = np.linalg.inv(T_dst) @ H_n @ T_src
	return H / H[2,2]


def task_2():
	M = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
	w_true = np.array([0.35, -0.55, 0.25])
	R_true = rodrigues(w_true)
	t_true = np.array([0.02, -0.01, 0.9])

	img = project_plane(M, R_true, t_true)

	cond_A = np.linalg.cond(make_A(M, img))
	src_n, _ = normalize_pts(M)
	dst_n, _ = normalize_pts(img)

	cond_N = np.linalg.cond(make_A(src_n, dst_n))
	
	print(f'{YELLOW}Проверка обусловленности матрицы{CLS}')
	print(f'Без нормализации\t{GREEN}{cond_A:.2f}{CLS}')
	print(f'С нормализацией\t\t{GREEN}{cond_N:.2f}{CLS}')
	print(f'Прирост равен\t\t{GREEN}{(cond_A / cond_N):.2f}{CLS}')
	print(f'Проверка прироста в {GREEN}100{CLS} раз\t\t=> {check(cond_N < (cond_A / 100))}')
	print(f'Проверка прироста с эталоном в {GREEN}23000{CLS}\t=> {check((cond_A / cond_N) >= 23000)}')

	M_gauss = rng.uniform(-.1, .1, (30, 2))
	img_gauss = project_plane(M_gauss, R_true, t_true)
	res_raw, res_norm = [], []

	for i in range(300):
		n = img_gauss + rng.normal(0, 1.0, img_gauss.shape)
		res_raw += [np.sqrt(np.mean((apply_H(dlt_homography(M_gauss, n), M_gauss) - img_gauss) ** 2))]
		res_norm += [np.sqrt(np.mean((apply_H(dlt_normalized(M_gauss, n), M_gauss) - img_gauss) ** 2))]

	print(f'\n{YELLOW}Проверка невязки с шумом в 1 пиксель{CLS}')
	print(f'Без нормализации:\t{np.mean(res_raw):.3f}')
	print(f'С нормализацией:\t{np.mean(res_norm):.3f}')

#endif

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