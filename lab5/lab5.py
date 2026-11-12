import sys

import numpy as np
import sympy as sp
import time
import matplotlib.pyplot as plt

import colors as c

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

def check(expr):
	return f"{c.style(c.bold, c.green)}OK{c.style(c.clear)}" if expr else "{c.style(c.bold, c.red)}FAIL{c.style(c.clear)}"

def normalize(v):
	return v / np.linalg.norm(v)

# параметры UR5: (a, alpha, d) для шести звеньев, метры и радианы
UR5 = [(0.0,      np.pi / 2,  0.089159),
       (-0.425,   0.0,        0.0),
       (-0.39225, 0.0,        0.0),
       (0.0,      np.pi / 2,  0.10915),
       (0.0,     -np.pi / 2,  0.09465),
       (0.0,      0.0,        0.0823)]

#region TASK 1

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

def dh_old(a, alpha, d, theta, lib=np):
	return R_z(theta, lib) @ T_z(d, lib) @ T_x(a, lib) @ R_x(alpha, lib)

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
	print(f"{c.style(c.yellow)}Используется {c.style(c.green, c.bold)}{lib.__name__}{c.style(c.clear)}")
	T = dh(0.5, 0, 0, lib.pi/2, lib)
	R = T[:3, :3]
	t = T[:3, 3]

	if lib is sp:
		R = sp.Matrix(R)

	p_exp = np.array([0., 0.5, 0.])
	print(f"Ожидаемое значение {p_exp}, получено {t}")
	
	det_func = np.linalg.det if lib is np else sp.det
	det_R = det_func(R)
	print(f"{c.style(c.bold)}R{c.style(c.clear)} = {R}")

	print(f"{c.style(c.bold)}det(R){c.style(c.clear)} = {det_R}\t\t=> {check(ue_is_nearly_equal(det_R, 1))}")
	
	norm_func = np.linalg.norm if lib is np else sp.Matrix.norm
	difference = norm_func(R.T @ R - lib.eye(3))
	print(f"{c.style(c.bold)}||R.T @ R - E||{c.style(c.clear)} = {difference}\t=> {check(ue_is_nearly_zero(difference))}")
	print()

def task_1():
	__task_1(np)
	__task_1(sp)

#endregion
#region TASK 2

def task_2():
	l1, l2, l3, q1, q2, q3 = sp.symbols('l1 l2 l3 q1 q2 q3', real=True)

	T = dh(l1, 0, 0, q1, sp) @ dh(l2, 0, 0, q2, sp) @ dh(l3, 0, 0, q3, sp)
	x_sym = sp.simplify(T[0,3])
	y_sym = sp.simplify(T[1,3])
	print(f'{c.style(c.yellow)}Вычислено с помощью SymPy{c.style(c.clear)}')
	print('x =', x_sym)
	print('y =', y_sym)

	# x связан с cos
	# y связан с sin
	x_ref = l1*sp.cos(q1) + l2*sp.cos(q1 + q2) + l3*sp.cos(q1 + q2 + q3)
	y_ref = l1*sp.sin(q1) + l2*sp.sin(q1 + q2) + l3*sp.sin(q1 + q2 + q3)
	
	print(f'\n{c.style(c.yellow)}Референсные формулы из шаблона{c.style(c.clear)}')
	print('x_ref =', x_ref)
	print('y_ref =', y_ref)

	chck_x = sp.simplify(x_sym - x_ref)
	chck_y = sp.simplify(y_sym - y_ref)

	print(f'\n{c.style(c.yellow)}Сверка результатов. Должно равняться нулю{c.style(c.clear)}')
	print(f'x - x_ref = {chck_x}\t=> {check(float(chck_x) == 0)}')
	print(f'y - y_ref = {chck_y}\t=> {check(float(chck_y) == 0)}')

	x_rot = T[0,0] # должно быть cos
	y_rot = T[1,0] # должно быть sin
	orient = sp.atan2(y_rot, x_rot) # atan2(sin(a), cos(a))

	print(f'\n{c.style(c.yellow)}Ориентация схвата{c.style(c.clear)}')
	print(f'x = {sp.simplify(x_rot)}')
	print(f'y = {sp.simplify(y_rot)}')
	print(f'ориентация схвата = {sp.simplify(orient)}')
	
#endregion

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
			print(f'{c.style(c.bold, c.yellow)}Задание %d{c.style(c.clear)}' % (i+1))
			LAB_TASKS[i]()
			print('\n')

	if len(sys.argv) == 2:
		id = int(sys.argv[1])
		print(f'{c.style(c.bold, c.yellow)}Задание %d{c.style(c.clear)}' % id)
		LAB_TASKS[id - 1]()
