import sys
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import itertools

# промежуток для рандомных чисел векторов сдвига
SHIFT_MAX: int = 5
SHIFT_MIN: int = -SHIFT_MAX
SEED = 0

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(seed=SEED)
random.seed(SEED)

# константа как в UE5
SMALL_NUMBER = 1e-8

def rand_num(MIN=0, MAX=10):
	return (random.random() % MAX) - MIN

# ue_is_nearly_zero(val: float) -> bool -- копирка bool IsNearlyZero(float, float) из UE5
def ue_is_nearly_zero(val: float) -> bool:
	return abs(val) <= SMALL_NUMBER

def ue_is_nearly_pi(val: float) -> bool:
	return abs(val - np.pi) <= 1e-4

def normalize(v):
	return v / np.linalg.norm(v)

# глобальные константы
RES = (1920, 1080)
CX = RES[0] / 2
CY = RES[1] / 2

MATRIX_WIDTH = 36 # обычный размер матрицы full-frame камеры 
DEFAULT_OPTIC_FOCUS = 50

# для удобного расчёта цифрового фокуса
def focus_mm_to_pixel(matrix_width=MATRIX_WIDTH, image_width=RES[0], focus_mm=DEFAULT_OPTIC_FOCUS):
	return focus_mm / ((matrix_width / image_width) / 2)

#region TASK 1

# f- фокус
# cx - центр в OX
# cy - центр в OY
def intrinsics(f: int|float, cx: int, cy: int) -> np.array:
	return np.array(
		[[f, 0, cx],
		[0, f, cy],
		[0, 0, 1.]])

def look_at(eye: np.asarray, target: np.asarray, up: np.asarray) -> tuple(np.asarray, np.asarray):
	z = target - eye; z = normalize(z)		# z - имеет направление от eye до target
	x = np.cross(z, up); x = normalize(x)	# x перпндикулярен и z, и верху
	y = np.cross(z, x)					# y - оставшийся базисный вектор

	# из трёх векторов получаем матрицу поворота
	R = np.stack([x,y,z])
	t = -R @ eye
	
	return R, t

def project(K, R, t, X):
	X_cam = X @ R.T + t
	z = X_cam[:, 2]
	p = X_cam @ K.T
	return p[:, :2] / p[:, 2:3], z

def task_1():
	f = focus_mm_to_pixel()
	cx, cy = CX, CY
	K = intrinsics(f, cx, cy)
	eye    = np.array([4., -3., 2.])
	target = np.array([0.,  0., 0.5])
	up     = np.array([0., 0., 1.])
	R, t = look_at(eye, target, up)
	print(R, t, sep='\n')
	print(np.linalg.det(R))
	print(np.linalg.norm(R.T @ R - np.eye(3)))

#endregion

#region TASK 2

def task_2():
	f = focus_mm_to_pixel()
	cx, cy = CX, CY

	K = intrinsics(f, cx, cy)
	eye    = np.array([10., 10, 2.5])
	target = np.array([0.,  0., 0])
	up     = np.array([0., 0., 1.])
	R, t = look_at(eye, target, up)

	vertices_coords = [
		(1., 0.5, 1), (-1., 0.5, 1), (1., 3.5, 1),
		(-1., 3.5, 1), (1., 0.5, 2), (-1., 0.5, 2),
		(1., 3.5, 2), (-1., 3.5, 2),
	]
	vertices_count = len(vertices_coords)
	vertices_coords = np.array(vertices_coords)

	indices = [
		(0,1), (0,2), (0,4),
		(2,3), (3,1), (3,7),
		(4,5), (2,6), (1,5),
		(4,6), (7,5), (7,6)
	]
	
	uv, z = project(K, R, t, vertices_coords)
	fig, ax = plt.subplots()
	for i, j in indices:
		ax.plot(*zip(uv[i], uv[j]), 'b-')
	ax.plot(cx, cy, 'r+', ms=12)
	ax.invert_yaxis()
	ax.set_aspect('equal')
	plt.show()

#endregion

#region TASK 3

def distort(xn, k1, k2, p1 = 0, p2 = 0):
	x = xn[..., 0]
	y = xn[..., 1]

	# получение квадрата радиуса по формуле
	r2 = x*x + y+y

	# по формуле для xd
	xd = x * (1 + k1*r2 + k2*r2*r2) + 2*p1*x*y + p2*(r2 + 2*x*x)
	xy = y * (1 + k1*r2 + k2*r2*r2) + p1*(r2 + 2*y*y) + 2*p2*x*x

	return np.stack([xd, xy], -1)
	

def undistort(xd, k1, k2, p1=0, p2=0, iterations=50):
	x = xd.copy()

	for i in range(iterations):
		# вычитаем смещение при искажении
		# чем больше итераций, тем ближе результат к значениям до искажения
		x -= distort(x, k1, k2, p1, p2) - xd
	
	return x

def task_3():

	F = focus_mm_to_pixel(36, RES[0], 58)
	print('focus =', F)
	max_x = RES[0] / F
	max_y = RES[1] / F

	print('max_x =', max_x, 'max_y =', max_y)

	ls_x = np.linspace(-max_x, max_x, 9)
	ls_y = np.linspace(-max_y, max_x, 9)
	gx, gy = np.meshgrid(ls_x, ls_y)
	x = np.stack([gx, gy], -1).reshape(-1, 2)
	
	k1 = -0.28
	k2 = 0.09
	p1 = 0.0
	p2 = 0.05

	iter_count = 30

	xd = distort(x, k1, k2, p1, p2)
	xd2 = distort(x, -k1, k2, p1, p2)
	xr = undistort(xd, k1, k2, p1, p2, iterations=iter_count)

	val = np.linalg.norm(undistort(distort(x, k1, k2), k1, k2, iterations=iter_count)) - np.linalg.norm(x)
	res = '\x1b[32;1mOK\x1b[0m' if abs(val) < 1e-12 else "\x1b[31;1mFAILED\x1b[0m"
	print(f"||undistort(distort(x)) - x|| = {val}. {res}\nКоличество итераций = {iter_count}, необходимо для удовлетворения условию - 20")

	fig, axs = plt.subplots(1, 4, figsize=(15, 6))
	for ax, P, title in zip(axs, (x, xd, xd2, xr), ("изначальный вариант", "искажённый k1=-0.28", "искажённый k1=+0.28", "восстановленный")):
		G = P.reshape(9, 9, 2)
		for i in range(9):
			ax.plot(G[i, :, 0], G[i, :, 1], 'b-')
			ax.plot(G[:, i, 0], G[:, i, 1], 'b-')
		ax.set_aspect('equal')
		ax.set_title(title)
	
	plt.show()

#endregion

#region TASK 4

def stereo_project(P, f, B, cx):
	u_L = f * P[:, 0] / P[:, 2] + cx
	u_R = f * (P[:, 0] - B) / P[:, 2] + cx
	return u_L, u_R

def calc_z(f, B, d):
	return f * B / d


def task_4():
	B = 0.12
	f = 800
	cx = 320.

	z_arr = np.array([0.5, 1, 2, 4, 8, 16])

	P  = np.stack([rand_num(0, 1) * np.ones_like(z_arr), rand_num(0, 1) * np.ones_like(z_arr), z_arr], 1)
	uL, uR = stereo_project(P, f, B, cx)
	d = uL - uR
	z_rec = calc_z(f, B, d)

	print('диспаратность d =', d)
	print('востановленные через формулу значения', z_rec)


#endregion
#region TASK 4
def task_5(): pass
#endregion
#region TASK 4
def task_6(): pass
#endregion
#region TASK 4
def task_7(): pass
#endregion


#region MAIN

LAB_TASKS = [task_1, task_2, task_3, task_4, task_5, task_6, task_7]

# if __name__ == '__main__':
# 	task_4()

# int main(int argc, char** argv) ;P
if __name__ == '__main__':
	if len(sys.argv) == 1:
		for i in range(len(LAB_TASKS)):
			print('\x1b[1;33mЗадание %d\x1b[0m' % (i+1))
			LAB_TASKS[i]()
			print('\n')

	if len(sys.argv) == 2:
		try:
			id = int(sys.argv[1])
			print('\x1b[1;33mЗадание %d\x1b[0m' % id)
			LAB_TASKS[id - 1]()
		except:
			sys.exit(0)

#endregion