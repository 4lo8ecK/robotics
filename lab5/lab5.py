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

RED = c.style(c.red)
GREEN = c.style(c.green)
YELLOW = c.style(c.yellow)
BOLD = c.style(c.bold)
CLS = c.style(c.clear)

def red(val) -> str: return f'{RED}{val}{CLS}'
def green(val) -> str: return f'{GREEN}{val}{CLS}'
def yellow(val) -> str: return f'{YELLOW}{val}{CLS}'
def bold(val) -> str: return f'{BOLD}{val}{CLS}'

def check(expr):
	return f'{c.style(c.bold, c.green)}OK{CLS}' if expr else f'{c.style(c.bold, c.red)}FAIL{CLS}'

# параметры UR5: (a, alpha, d) для шести звеньев, метры и радианы
def UR5(lib=np):
	return [(0.0,      lib.pi / 2,  0.089159),
			(-0.425,   0.0,        0.0),
			(-0.39225, 0.0,        0.0),
			(0.0,      lib.pi / 2,  0.10915),
			(0.0,     -lib.pi / 2,  0.09465),
			(0.0,      0.0,        0.0823)]

#region TASK 1

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
	print(f'{BOLD}||R.T @ R - E||{CLS} = {difference}\t=> {check(ue_is_nearly_zero(difference))}')
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
	print(f'{YELLOW}Вычислено с помощью SymPy{CLS}')
	print('x =', x_sym)
	print('y =', y_sym)

	# x связан с cos
	# y связан с sin
	x_ref = l1*sp.cos(q1) + l2*sp.cos(q1 + q2) + l3*sp.cos(q1 + q2 + q3)
	y_ref = l1*sp.sin(q1) + l2*sp.sin(q1 + q2) + l3*sp.sin(q1 + q2 + q3)
	
	print(f'\n{YELLOW}Референсные формулы из шаблона{CLS}')
	print('x_ref =', x_ref)
	print('y_ref =', y_ref)

	chck_x = sp.simplify(x_sym - x_ref)
	chck_y = sp.simplify(y_sym - y_ref)

	print(f'\n{YELLOW}Сверка результатов. Должно равняться нулю{CLS}')
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
	print(f'num\t{GREEN}{fk_num_time} мкс{CLS}')
	print(f'sym\t{GREEN}{fk_sym_time} мкс{CLS}')
	print(f'numsym\t{GREEN}{(f'{(fk_sym_time / fk_num_time):.2f}')}{CLS}')

	f_lamdify = sp.lambdify(angles_sp, forward_kinematic(sp, angles_sp), 'numpy', cse=False)
	f_cse = sp.lambdify(angles_sp, forward_kinematic(sp, angles_sp), 'numpy', cse=True)

	angles_np = rng.uniform(-np.pi, np.pi, 6)

	A = forward_kinematic(np, angles_np)
	B = np.array(f_lamdify(*angles_np), dtype=float)
	C = np.array(f_cse(*angles_np), dtype=float)

	delta_lambda = np.abs(A - B).max()
	delta_cse = np.abs(A - C).max()
	print(yellow('\nПроверка ошибок'))
	print(f'num - lambdify\t{GREEN}{delta_lambda:.2e}{CLS}\t=> {check(ue_is_nearly_zero(delta_lambda))}')
	print(f'num - cse\t{GREEN}{delta_cse:.2e}{CLS}\t=> {check(ue_is_nearly_zero(delta_cse))}')

	T_sp: sp.Matrix = sp.Matrix(forward_kinematic(sp, angles_sp))
	sub = dict(zip(angles_sp, angles_np))
	time_subs = timer(lambda: np.array(T_sp.subs(sub).evalf(), dtype=float))
	time_lambda = timer(lambda: f_lamdify(*angles_np))
	time_cse = timer(lambda: f_cse(*angles_np))
	time_loop = timer(lambda: forward_kinematic(np, angles_np))

	print(yellow('\nЗамер скоростей работы'))
	print(f'{c.style(c.bold)}{'метод':16s}{'мс':10s}{'ускорение'}{CLS}')
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
#region TASK 4

# площадь кольца, указанного через l1 и l2
# где r1 = |l1 - l2|, r2 = l1 + l2
# pi*(l1+l2)^2 - pi*(l1-l2)^2 => pi*((l1+l2)^2 - (l1-l2)^2) => 4*pi*l1*l2
def calc_ring_area(l1: float, l2: float) -> float:
	return np.pi * (4 * l1 * l2)

def monte_carlo_area(l1: float, l2: float, points_count: int) -> float:
	r_out = l1+l2
	r_in = abs(l1-l2)
	
	x = np.random.uniform(-r_out, r_out, points_count)
	y = np.random.uniform(-r_out, r_out, points_count)

	dist_square = x*x + y*y
	inside = (dist_square >= r_in*r_in) &  (dist_square <= r_out*r_out)
	return 4*r_out*r_out * np.mean(inside)

	
def task_4():
	steps = 30
	l1, l2 = 0.5, 0.35
	print(f'l1 = {l1}, l2 = {l2}')
	print(f'Площадь кольца по формуле:\t{calc_ring_area(0.5, 0.35):.6f}')

	numbers = np.logspace(2, 6, steps).astype(int)
	errs = [np.mean([abs(monte_carlo_area(l1, l2, N) - calc_ring_area(l1,l2)) for _ in range(steps)]) for N in numbers]
	slope = np.polyfit(np.log10(numbers), np.log10(errs), 1)[0]
	print(f'Наклон = {slope}')

	plt.loglog(numbers, errs, 'o-', label='ошибка (среднее по 30 запускам)')
	plt.loglog(numbers, errs[0] * (numbers / numbers[0])**-0.5, '--', label='N^(-1/2)')
	plt.grid(visible=True, which='both')
	plt.show()

#endregion TASK 5

def voxel_volume(points, h):
	indices = np.floor(points / h).astype(int)
	found = len(np.unique(indices, axis=0))
	return found * (h ** 3)

def task_5():
	angles_sp = sp.symbols('q1 q2 q3 q4 q5 q6', real=True)
	f_cse = sp.lambdify(angles_sp, forward_kinematic(sp, angles_sp), 'numpy', cse=True)

	N = 20000
	angles = rng.uniform(-np.pi, np.pi, (N, 6))
	points = np.array([np.array(f_cse(*a), dtype=float)[:3,3] for a in angles])

	print(yellow('Проверка зависимости объема от значения h'))
	for h in (0.08, 0.05, 0.03):
		print(f'h = {h}\tV = {voxel_volume(points, h):.4f}')

	print(yellow('\nПроврека зависимости объема от числа точек при h = 0.05'))
	for n in np.linspace(100, N, 5, dtype=int):
		print(f'n = {n}\tV = {voxel_volume(points[:n], 0.05):.8f}')

	

#endregion
def task_6(): pass
def task_7(): pass

LAB_TASKS = [task_1, task_2, task_3, task_4, task_5, task_6, task_7]

# int main(int argc, char** argv) ;P
if __name__ == '__main__':
	try:	
		# если нет аргументов, выполняются все задания
		if len(sys.argv) == 1:
			for i in range(len(LAB_TASKS)):
				print(f'{c.style(c.bold, c.yellow)}Задание {i + 1}{CLS}')
				LAB_TASKS[i]()
				print('\n')
		# выполняется задание, указанное в аргументе
		if len(sys.argv) == 2:
			id = int(sys.argv[1])
			print(f'{c.style(c.bold, c.yellow)}Задание {id}{CLS}')
			LAB_TASKS[id - 1]()

	except KeyboardInterrupt:
		sys.exit(0)