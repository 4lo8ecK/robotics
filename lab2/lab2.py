import sys
import time

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

def normalize(v):
	return v / np.linalg.norm(v)


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
		n = normalize(n)
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
	q = normalize(q)
	
	return q if q[0] >= 0 else -q

def quat_to_R(q):
	q = normalize(q)
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
	conds = []
	print('\t|   тангаж\t|   cond\t|   det\t\t|   cos\t\t|')
	for i in [0, 30, 60, 80, 89.9, 89.99]:
		E = euler_rate_matrix((0, np.radians(i), 0))
		cond = np.linalg.cond(E)
		conds += [cond]
		print('\t|   %.2f\t|   %.3f\t|   %.2e\t|   %.2e\t|' % (i, cond, abs(np.linalg.det(E)), np.radians(i)))

	pp = np.linspace(0, 89.99, 1000)
	plt.semilogy(pp, [np.linalg.cond(euler_rate_matrix([0.3, np.radians(p), 0.5])) for p in pp])
	plt.xlabel('тангаж, °')
	plt.ylabel('cond(E)')
	plt.grid(True)
	plt.show()

#endregion

#region TASK 5

def slerp(q0, q1, t):
	q = np.dot(q0, q1)
	
	if q < 0: q1, q = -q1, -q
	if q > 0.999:
		q = q0 + t * (q1 - q0)
		return normalize(q)
	
	omega = np.arccos(q)
	return (np.sin((1 - t) * omega) * q0 + np.sin(t * omega) * q1) / np.sin(omega)

def task_5():
	ea, eb = np.radians([0,0,0]), np.radians([150,60,-100])
	Ra, Rb = euler_zyx_to_R(ea), euler_zyx_to_R(eb)
	qa, qb = R_to_quat(Ra), R_to_quat(Rb)
	N = 400; ts = np.linspace(0, 1, N); dt = ts[1]-ts[0]

	R_sl = [quat_to_R(slerp(qa, qb, t)) for t in ts]
	R_eu = [euler_zyx_to_R(ea + t*(eb-ea)) for t in ts]
	speed = lambda Rl: np.array([np.linalg.norm(so3_log(Rl[i+1] @ Rl[i].T))/dt for i in range(len(Rl)-1)])
	w_sl, w_eu = speed(R_sl), speed(R_eu)

	plt.plot(w_sl, label='SLERP')
	plt.plot(w_eu, label='углы Эйлера')
	plt.xlabel('шаг'); plt.ylabel('|ω|, рад/с')
	plt.legend()
	plt.grid(True)
	plt.show()

#endregion

#region TASK 6

# умножение кватернионов
def mult_q(a, b):
	w1,x1,y1,z1 = a
	w2,x2,y2,z2 = b
	return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def task_6():
	w = np.array([0.3, -0.5, 0.8])
	wn = np.linalg.norm(w)
	dt = 1e-3
	n = 200000
	dR = so3_exp(w*dt)
	dq = np.concatenate([[np.cos(wn*dt / 2)], w / wn * np.sin(wn*dt / 2)])

	R = np.eye(3)
	q1 = np.array([1,0,0,0])
	q2 = np.array([1,0,0,0])

	check = set(np.unique(np.logspace(0, np.log10(n), 40).astype(int)))
	res = []
	for i in range(1, n+1):
		R  = R @ dR
		q1 = mult_q(q1, dq)
		q2 = mult_q(q2, dq)
		q2 = normalize(q2)

		if i in check:
			res.append((i, np.linalg.norm(R.T @ R - np.eye(3)),
						abs(np.linalg.norm(q1) - 1),
						abs(np.linalg.norm(q2) - 1)))
	res = np.array(res)

	plt.loglog(res[:,0], res[:,1] + 1e-18, label='матрица')
	plt.loglog(res[:,0], res[:,2] + 1e-18, label='кватернион без нормировки')
	plt.loglog(res[:,0], res[:,3] + 1e-18, label='кватернион с нормировкой')
	plt.xlabel('шаг')
	plt.ylabel('ошибка')
	plt.legend()
	plt.grid(True)
	plt.show()

#endregion

#region TASK 7

# ортогонализация матрицы
def ort_svd(mat):
    u, s, vt = np.linalg.svd(mat)
    det_corr = np.linalg.det(u @ vt)
    diag_corr = [1.0, 1.0, det_corr]
    return u @ np.diag(diag_corr) @ vt

# нормализация кватерниона
def norm_quat(quat):
    return quat / np.linalg.norm(quat)

def task_7():
	matrix = rand_R()
	quat = rand_vector(size=4)

	iter_count = 20000

	t_start = time.perf_counter()
	for i in range(iter_count):
		ort_svd(matrix)
	dt_svd = (time.perf_counter() - t_start) / iter_count

	t_start = time.perf_counter()
	for i in range(iter_count):
		norm_quat(quat)
	dt_quat = (time.perf_counter() - t_start) / iter_count

	t_svd = dt_svd * 1e6
	t_quat = dt_quat * 1e6

	print(f'svd-ортогонализация \t- {t_svd}\nкватернион \t\t- {t_quat}')

#endregion

LAB_TASKS = [task_1, task_2, task_3, task_4, task_5, task_6, task_7]

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