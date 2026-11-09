import sys

import numpy as np
import sympy as sp
import time
import matplotlib.pyplot as plt

# промежуток для рандомных чисел векторов сдвига
SHIFT_MAX: int = 5
SHIFT_MIN: int = -SHIFT_MAX
SEED = 0

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(seed=SEED)

# константа как в UE5
SMALL_NUMBER = 1e-12

# ue_is_nearly_zero(val: float) -> bool -- копирка bool IsNearlyZero(float, float) из UE5
def ue_is_nearly_zero(val: float) -> bool:
	return abs(val) <= SMALL_NUMBER

def ue_is_nearly_equal(a: float, b: float) -> bool:
	return abs(a - b) <= SMALL_NUMBER

def ue_is_nearly_pi(val: float) -> bool:
	return abs(val - np.pi) <= 1e-4

def normalize(v):
	return v / np.linalg.norm(v)

# параметры UR5: (a, alpha, d) для шести звеньев, метры и радианы
UR5 = [(0.0,      np.pi / 2,  0.089159),
       (-0.425,   0.0,        0.0),
       (-0.39225, 0.0,        0.0),
       (0.0,      np.pi / 2,  0.10915),
       (0.0,     -np.pi / 2,  0.09465),
       (0.0,      0.0,        0.0823)]



def R_x(alpha, lib=np):
	c = lib.cos(alpha)
	s = lib.sin(alpha)
	res = [
		[1, 0, 0, 0],
		[0, c, -s, 0],
		[0, s, c, 0],
		[0, 0, 0, 1]
	]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)

def R_z(alpha, lib=np):
	c = lib.cos(alpha)
	s = lib.sin(alpha)
	res = [
		[c, -s, 0, 0],
		[s, c, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)

def T_x(a, lib=np):
	res = [
		[1, 0, 0, a],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)

def T_z(a, lib=np):
	res = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, a],
		[0, 0, 0, 1]
	]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)


def dh(a, alpha, d, theta, lib=np):
	# $$T_i = R_z(\theta_i)\, T_z(d_i)\, T_x(a_i)\, R_x(\alpha_i).$$
	# res = R_z(theta, lib) @ T_z(d, lib) @ T_x(a, lib) @ R_x(alpha, lib)
	ct, st = lib.cos(theta), lib.sin(theta)
	ca, sa = lib.cos(alpha), lib.sin(alpha)
	res = [[ct, -st*ca,  st*sa, a*ct],
			[st,  ct*ca, -ct*sa, a*st],
			[0,   sa,     ca,    d   ],
			[0,   0,      0,     1   ]]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)


def __task_1(lib):
	print(f"Используется {lib.__name__}")
	T = dh(0.5, 0, 0, lib.pi/2, lib)
	R = T[:3, :3]
	t = T[:3, 3]

	if lib is sp:
		R = sp.Matrix(R)

	p_exp = np.array([0., 0.5, 0.])
	print(f"Ожидаемое значение {p_exp}, получено {t}")
	
	det_func = np.linalg.det if lib is np else sp.det
	det_R = det_func(R)
	print(f"R = {R}")

	res_ok_no = "\x1b[32;1mOK\x1b[0m" if ue_is_nearly_equal(det_R, 1) else "\x1b[31;1mFAIL\x1b[0m"
	print(f"det(R) = {det_R} => {res_ok_no}")
	
	norm_func = np.linalg.norm if lib is np else sp.Matrix.norm

	difference = norm_func(R.T @ R - lib.eye(3))
	res_ok_no = "\x1b[32;1mOK\x1b[0m" if ue_is_nearly_zero(difference) else "\x1b[31;1mFAIL\x1b[0m"
	print(f"||R.T @ T - E|| = {difference} => {res_ok_no}")
	print()

def task_1():
	__task_1(np)
	__task_1(sp)
	


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
