import sys

import numpy as np
import matplotlib.pyplot as plt

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


def rotx(a):
	c, s = np.cos(a), np.sin(a)
	return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(a):
	c, s = np.cos(a), np.sin(a)
	return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(a):
	c, s = np.cos(a), np.sin(a)
	return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rand_vector(min: int | float = SHIFT_MIN, max: int | float = SHIFT_MAX, size: int = 3):
	return rng.uniform(low=min, high=max, size=size)

# скопировано из первой лабораторки (из random_transform)
def rand_R() -> np.asarray:
	a = rng.uniform(-np.pi, np.pi, size=3)
	return rotx(a[0]) @ roty(a[1]) @ rotz(a[2])

#region TASK 1

# формулу нашёл в интернете (vmath - https://vmath.ru/vf5/algebra2/skewsym)
def skew(w):
	x, y, z = w[0], w[1], w[2]
	return np.asarray(
		[[0, -z,  y],
		[ z,  0, -x],
		[-y,  x,  0]])

def unskew(K):
	return np.asarray([K[2,1], K[0,2], K[1,0]])

# по формуле родрига R = exp(w) = E + sin(a) * sw + (1 - cos(a)) * (sw @ sw) 
# a - угол, то же, что и длина вектора w
# sw - кососимметрическая матрица, полученная через skew(w)
# где E - единичная матрица
def so3_exp(w):
	a = np.linalg.norm(w)
	w = skew(w)

	# т.к. в формуле родрига w нужно разделить на угол:
	# когда a стремится к нулю
	if ue_is_nearly_zero(a):
		# a -> 0: sin(a) = a; cos(a) = 1 - a*a/2 => 1 - cos(a) = a*a/2 или = 2sin(a/2)^2
		# в таком случае получаем a*w/a, а не cos(a)*w/a
		# то же и для cos => (w@w)/2
		return np.eye(3) + w + 0.5 * (w @ w)
	# все остальные случаи
	w /= a
	return np.eye(3) + np.sin(a) * w + (1 - np.cos(a)) * (w @ w)

# a = arccos((tr R - 1) / 2)
# sw = (R - R.T) / 2sin(a)
def so3_log(R: np.asarray):
	a = np.arccos((np.trace(R) - 1) / 2)
	
	if ue_is_nearly_zero(a):
		# т.к. при a -> 0: sin(a) = a
		# это то же, что и a * unskew((trR - 1) / 2 * a)
		# a и a сокращаются и остаётся:
		return unskew((np.trace(R) - 1) / 2)

	if ue_is_nearly_pi(a):
		nnt = (R + np.eye(3)) / 2
		i = np.argmax(np.diag(nnt))
		n = nnt[:i] / np.sqrt(nnt[i,i])
		n /= np.linalg.norm(n)
		if np.dot((R - R.T) / 2, n) < 0: n = -n
		return a * n

	# общий случай (по формуле)
	return a * unskew((R - R.T) / (2 * np.sin(a)))

def task_1():
	err_count = 0
	err_max_delta = 0
	for i in range(300):
		R = rand_R()
		delta = np.linalg.norm(so3_exp(so3_log(R)) - R)
		if delta > err_max_delta: err_max_delta = delta
		if delta > 1e-9:
			err_count += 1
	print(f"||exp(log(R)) - R||\nmax delta = {err_max_delta}\nerr count = {err_count}")

#endregion

#region TASK 2

def R_to_quat(R):
	# нашёл в интернете формулу и нудно писал всё по ней

	t = np.trace(R)
	i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2], t]))

	# в случае w 
	if i == 3:
		s = 2 * np.sqrt(1 + t)
		w = s / 4
		x = (R[2, 1] - R[1, 2]) / s
		y = (R[0, 2] - R[2, 0]) / s
		z = (R[1, 0] - R[0, 1]) / s
	# в случае x
	elif i == 0:
		s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
		x = s / 4
		w = (R[2, 1] - R[1, 2]) / s
		y = (R[0, 1] + R[1, 0]) / s
		z = (R[0, 2] + R[2, 0]) / s
	# в случае y
	elif i == 1:
		s = 2 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
		y = s / 4
		w = (R[0, 2] - R[2, 0]) / s
		x = (R[0, 1] + R[1, 0]) / s
		z = (R[1, 2] + R[2, 1]) / s
	# в случае z
	else:
		s = 2 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
		z = s / 4
		w = (R[1, 0] - R[0, 1]) / s
		x = (R[0, 2] + R[2, 0]) / s
		y = (R[1, 2] + R[2, 1]) / s
	
	q = np.array([w, x, y, z])
	q /= np.linalg.norm(q)
	
	return q if q[0] >= 0 else -q

def quat_to_R(q):
	q /= np.linalg.norm(q)
	w, x, y, z = q[0], q[1], q[2], q[3]

	return np.array(
		[[1 - 2 * (y*y + z*z),	2 * (x*y - w*z),		2 * (x*z + w*y)],
		[2 * (x*y + w*z),		1 - 2 * (x*x + z*z),	2 * (y*z - w*x)],
		[2 * (x*z - w*y), 		2 * (y*z + w*x),		1 - 2 * (x*x + y*y)]])

def R_to_euler_zyx(R):
	return np.array(
		[np.atan2(R[1,0], R[0,0]),
		np.arcsin(-R[2,0]),
		np.atan2(R[2,1], R[2,2])])

def euler_zyx_to_R(a):
	return rotz(a[0]) @ roty(a[1]) @ rotx(a[2])


def task_2():
	quat_err_max_delta = 0
	quat_err_count = 0
	euler_err_max_delta = 0
	euler_err_count = 0
	
	for i in range(300):
		R = rand_R()
		quat_delta = np.linalg.norm(quat_to_R(R_to_quat(R)) - R)
		euler_delta = np.linalg.norm(euler_zyx_to_R(R_to_euler_zyx(R)) - R)

		if quat_delta > quat_err_max_delta: quat_err_max_delta = quat_delta
		if quat_delta >= 1e-15: quat_err_count += 1 
		
		if euler_delta > euler_err_max_delta: euler_err_max_delta = euler_delta
		if euler_delta >= 1e-10: euler_err_count += 1 

	print(f"Кватернионы:\n\tmax_err_delta = {quat_err_max_delta}\n\terr_count = {quat_err_count}\nУглы Эйлера:\n\tmax_err_delta = {euler_err_max_delta}\n\terr_count = {euler_err_count}")

#endregion

# region TASK 3

def task_3():
	q = rand_vector(size=4)
	print(f'Кватернион: q = {q}\n')
	print(f'R из q:\n{quat_to_R(q)}\n')
	print(f'R из -q:\n{quat_to_R(-q)}\n')
	print(f'разница между R из q и -q = {np.linalg.norm(quat_to_R(q) - quat_to_R(-q))}')

#endregion

#region TASK 4

def euler_rate_matrix(a):
	return np.array(
		[[0, -np.sin(a[0]), np.cos(a[1])*np.cos(a[0])],
		[0, np.cos(a[0]), np.cos(a[1])*np.sin(a[0])],
		[1, 0, - np.sin(a[1])]])

def task_4():
	print('\t|   тангаж\t|   cond\t|   det\t\t|   cos\t\t|')
	for i in [0, 30, 60, 80, 89.9, 89.99]:
		E = euler_rate_matrix((0, np.radians(i), 0))
		cond = np.linalg.cond(E)
		print('\t|   %.2f\t|   %.3f\t|   %.2e\t|   %.2e\t|' % (i, cond, abs(np.linalg.det(E)), np.radians(i)))

#endregion

if __name__ == '__main__':
	task_4()