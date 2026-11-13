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
SEED = 0

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(seed=SEED)

# константа как в UE5
SMALL_NUMBER = 1e-12

# ue_is_nearly_zero(val: float) -> bool -- копирка bool IsNearlyZero(float, float) из UE5
def ue_is_nearly_zero(val: float) -> bool: return abs(val) <= SMALL_NUMBER
def ue_is_nearly_equal(a: float, b: float) -> bool: return abs(a - b) <= SMALL_NUMBER
def ue_is_nearly_pi(val: float) -> bool: return abs(val - np.pi) <= 1e-4

def red(val) -> str: return f'{c.style(c.red)}{val}{c.style(c.clear)}'
def green(val) -> str: return f'{c.style(c.green)}{val}{c.style(c.clear)}'
def yellow(val) -> str: return f'{c.style(c.yellow)}{val}{c.style(c.clear)}'
def bold(val) -> str: return f'{c.style(c.bold)}{val}{c.style(c.clear)}'

def check(expr):
	return f'{bold(green('OK'))}' if expr else f'{bold(red('FAIL'))}'

# параметры UR5: (a, alpha, d) для шести звеньев, метры и радианы
def UR5(lib=np):
	return [(0.0,      lib.pi / 2,  0.089159),
			(-0.425,   0.0,        0.0),
			(-0.39225, 0.0,        0.0),
			(0.0,      lib.pi / 2,  0.10915),
			(0.0,     -lib.pi / 2,  0.09465),
			(0.0,      0.0,        0.0823)]

# region TASK 1

def R_x(alpha: float, lib=np) -> sp.Matrix|np.ndarray:
	c = lib.cos(alpha)
	s = lib.sin(alpha)
	res = [
		[1, 0, 0, 0],
		[0, c, -s, 0],
		[0, s, c, 0],
		[0, 0, 0, 1]
	]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)

def R_z(alpha, lib=np) -> sp.Matrix|np.ndarray:
	c = lib.cos(alpha)
	s = lib.sin(alpha)
	res = [
		[c, -s, 0, 0],
		[s, c, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)

def T_x(a, lib=np) -> sp.Matrix|np.ndarray:
	res = [
		[1, 0, 0, a],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)

def T_z(a, lib=np) -> sp.Matrix|np.ndarray:
	res = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, a],
		[0, 0, 0, 1]
	]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)

def dh_old(a, alpha, d, theta, lib=np) -> sp.Matrix|np.ndarray:
	return R_z(theta, lib) @ T_z(d, lib) @ T_x(a, lib) @ R_x(alpha, lib)

def dh(a, alpha, d, theta, lib=np) -> sp.Matrix|np.ndarray:
	# res = R_z(theta, lib) @ T_z(d, lib) @ T_x(a, lib) @ R_x(alpha, lib)
	ct, st = lib.cos(theta), lib.sin(theta)
	ca, sa = lib.cos(alpha), lib.sin(alpha)
	res = [[ct, -st*ca,  st*sa, a*ct],
			[st,  ct*ca, -ct*sa, a*st],
			[0,   sa,     ca,    d   ],
			[0,   0,      0,     1   ]]
	return sp.Matrix(res) if lib is sp else np.asarray(res, dtype=float)

def __task_1(lib=np):
	print(f'{yellow('Используется')} {green(lib.__name__)}')
	T = dh(0.5, 0, 0, lib.pi/2, lib)
	R = T[:3, :3]
	t = T[:3, 3]

	if lib is np:
		R = np.array(R)
	else:
		R = sp.Matrix(R)

	p_exp = np.array([0., 0.5, 0.])
	print(f'Ожидаемое значение {p_exp}, получено {t}')
	
	det_func = np.linalg.det if lib is np else sp.det
	det_R: float = float(det_func(R))
	print(f'{bold('R')} = {R}')

	print(f'{bold('det(R)')} = {det_R}\t\t=> {check(ue_is_nearly_equal(det_R, 1))}')
	
	norm_func = np.linalg.norm if lib is np else sp.Matrix.norm
	difference = norm_func(R.T @ R - lib.eye(3))
	print(f'{c.style(c.bold)}||R.T @ R - E||{c.style(c.clear)} = {difference}\t=> {check(ue_is_nearly_zero(difference))}')
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

	print(f'\n{yellow('Ориентация схвата')}')
	print(f'x = {sp.simplify(x_rot)}')
	print(f'y = {sp.simplify(y_rot)}')
	print(f'ориентация схвата = {sp.simplify(orient)}')
	
#endregion

#region TASK 3
# возвращает время выполнения функции n-ое количество раз в микросекундах
def timer(func: Callable, count: int = 1) -> int:
	t = time.perf_counter_ns()
	for i in range(count):
		func()
	return (time.perf_counter_ns() - t) // 1000

# численный и символьный способы построения кинематики
# т.е. через методы numpy и sympy
# 2 в 1
def forward_kinematic(lib: ModuleType = np, angles: Iterable|Any|None = None) -> sp.Matrix|np.ndarray:
	assert angles is not None, TypeError

	T = lib.eye(4)
	for (a, alpha, d), th in zip(UR5(np), angles):
		T = T @ dh(a, alpha, d, th, lib)
	return sp.Matrix(T) if lib is sp else np.array(T)


def task_3():
	angles_sp = sp.symbols('q1 q2 q3 q4 q5 q6', real=True)

	fk_num_time = timer(lambda: forward_kinematic(np, [1,0,6,0,0,0]))
	fk_sym_time = timer(lambda: forward_kinematic(sp, angles_sp))
	
	print(yellow('Скорость создания матриц кинематики'))
	print(f'num\t{c.style(c.green)}{fk_num_time} мкс{c.style(c.clear)}')
	print(f'sym\t{c.style(c.green)}{fk_sym_time} мкс{c.style(c.clear)}')
	print(f'numsym\t{c.style(c.green)}{(f'{(fk_sym_time / fk_num_time):.2f}')}{c.style(c.clear)}')

	f_lamdify = sp.lambdify(angles_sp, forward_kinematic(sp, angles_sp), 'numpy', cse=False)
	f_cse = sp.lambdify(angles_sp, forward_kinematic(sp, angles_sp), 'numpy', cse=True)

	angles_np = rng.uniform(-np.pi, np.pi, 6)

	A = forward_kinematic(np, angles_np)
	B = np.array(f_lamdify(*angles_np), dtype=float)
	C = np.array(f_cse(*angles_np), dtype=float)

	delta_lambda = np.abs(A - B).max()
	delta_cse = np.abs(A - C).max()
	print(yellow('\nПроверка ошибок'))
	print(f'num - lambdify\t{c.style(c.green)}{delta_lambda:.2e}{c.style(c.clear)}\t=> {check(ue_is_nearly_zero(delta_lambda))}')
	print(f'num - cse\t{c.style(c.green)}{delta_cse:.2e}{c.style(c.clear)}\t=> {check(ue_is_nearly_zero(delta_cse))}')

	T_sp: sp.Matrix = sp.Matrix(forward_kinematic(sp, angles_sp))
	sub = dict(zip(angles_sp, angles_np))
	time_subs = timer(lambda: np.array(T_sp.subs(sub).evalf(), dtype=float))
	time_lambda = timer(lambda: f_lamdify(*angles_np))
	time_cse = timer(lambda: f_cse(*angles_np))
	time_loop = timer(lambda: forward_kinematic(np, angles_np))

	print(yellow('\nЗамер скоростей работы'))
	print(f'{c.style(c.bold)}{'метод':16s}{'мс':10s}{'ускорение'}{c.style(c.clear)}')
	for name, time in [
			('subs + evalf', time_subs),
			('lambda', time_lambda),
			('cse', time_cse),
			('numpy-loop', time_loop)]:
		print(f'{name:16s}{str(time):10s}{(time_subs/time):.2f}')

	N = 20000
	Q = rng.uniform(-np.pi, np.pi, (N, 6))
	P = np.array([np.array(f_cse(*q), dtype=float)[:3, 3] for q in Q])

	reach = np.linalg.norm(P, axis=1).max()
	upper = sum(np.hypot(a, d) for a, i, d in UR5(np))
	lower = abs(UR5(np)[1][0]) + abs(UR5(np)[2][0])

	print(yellow('\nОблако точек'))
	print(f'a2 + a3\t\t\t{lower:.4f} м')
	print(f'измеренный вылет\t{reach:.4f} м')
	print(f'оценка сверху\t\t{upper:.4f} м')
	print(f'лежит между границами => {check(lower < reach < upper)}')	
#endregion

def task_4(): pass
def task_5(): pass
def task_6(): pass
def task_7(): pass

LAB_TASKS = [task_1, task_2, task_3, task_4, task_5, task_6, task_7]

# int main(int argc, char** argv) ;P
if __name__ == '__main__':
	try:	
		# если нет аргументов, выполняются все задания
		if len(sys.argv) == 1:
			for i in range(len(LAB_TASKS)):
				print(f'{c.style(c.bold, c.yellow)}Задание {i + 1}{c.style(c.clear)}')
				LAB_TASKS[i]()
				print('\n')
		# выполняется задание, указанное в аргументе
		if len(sys.argv) == 2:
			id = int(sys.argv[1])
			print(f'{c.style(c.bold, c.yellow)}Задание {id}{c.style(c.clear)}')
			LAB_TASKS[id - 1]()

	except KeyboardInterrupt:
		sys.exit(0)