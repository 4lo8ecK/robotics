import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import itertools

# промежуток для рандомных чисел векторов сдвига
SHIFT_MAX: int = 5
SHIFT_MIN: int = -SHIFT_MAX
SEED = None

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(seed=SEED)

# константа как в UE5
SMALL_NUMBER = 1e-8

# ue_is_nearly_zero(val: float) -> bool -- копирка bool IsNearlyZero(float, float) из UE5
def ue_is_nearly_zero(val: float) -> bool:
	return abs(val) <= SMALL_NUMBER

def ue_is_nearly_pi(val: float) -> bool:
	return abs(val - np.pi) <= 1e-4

def normalize(v):
	return v / np.linalg.norm(v)


#region TASK 1

# intrinsics(f, cx, cy)`, `look_at(eye, target, up)` и `project(K, R, t, X)

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
	f, cx, cy = 50., 320., 240.
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

	f, cx, cy = 0.35, 320., 240.
	K = intrinsics(f, cx, cy)
	eye    = np.array([10., 10, 2.5])
	target = np.array([0.,  0., 0])
	up     = np.array([0., 0., 1.])
	R, t = look_at(eye, target, up)

	vertices_coords = [
		(1., 0.5, 1),
		(-1., 0.5, 1),
		(1., 3.5, 1),
		(-1., 3.5, 1),
		(1., 0.5, 2),
		(-1., 0.5, 2),
		(1., 3.5, 2),
		(-1., 3.5, 2),
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
		ax.plot(*zip(uv[i], uv[j]), 'b-')   # линия между двумя спроецированными вершинами
	ax.plot(cx, cy, 'r+', ms=12)            # главная точка
	ax.invert_yaxis()                       # ось v вниз, иначе картинка вверх ногами
	ax.set_aspect('equal')
	plt.show()

#endregion



if __name__ == '__main__':
	task_2()