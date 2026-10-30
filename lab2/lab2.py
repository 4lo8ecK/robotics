import sys

import numpy as np
import matplotlib.pyplot as plt

# промежуток для рандомных чисел векторов сдвига
SHIFT_MAX: int = 5
SHIFT_MIN: int = -SHIFT_MAX
SEED = 0

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(seed=SEED)

# константа как в UE5
SMALL_NUMBER = 1e-8

# ue_is_nearly_equal(val: float) -> bool -- копирка bool IsNearlyEqual(float, float) из UE5
def ue_is_nearly_equal(a: float, b: float) -> bool:
	return abs(a - b) <= SMALL_NUMBER

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


#endregion

if __name__ == '__main__':
	task_1()