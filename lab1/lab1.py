import sys

import numpy as np
import matplotlib.pyplot as plt

# промежуток для рандомных чисел векторов сдвига
SHIFT_MAX: int = 5
SHIFT_MIN: int = -SHIFT_MAX
SEED = 0

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(seed=SEED)

def rotx(a):
	return np.asarray(
		[[1, 0, 0],
		[0, np.cos(a), -np.sin(a)],
		[0, np.sin(a), np.cos(a)]],
		dtype=float)
def roty(a):
	return np.asarray(
		[[np.cos(a), 0, np.sin(a)],
		[0, 1, 0],
		[-np.sin(a), 0, np.cos(a)]],
		dtype=float)
def rotz(a):
	return np.asarray(
		[[np.cos(a), -np.sin(a), 0],
		[np.sin(a), np.cos(a), 0],
		[0, 0, 1]],
		dtype=float)

class Transform:
	"""Жёсткое преобразование как элемент SE(3), хранимое матрицей 4x4."""

	def __init__(self, matrix=None):
		self.M = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)
		if self.M.shape != (4, 4):
			raise palueError('нужна матрица 4x4, получено %s' % (self.M.shape,))

	@classmethod
	def from_Rt(cls, R, t):
		if R is None: R = np.eye(4)
		mtx = np.eye(4)
		mtx[:3, :3] = R[:3, :3]
		mtx[:3, 3] = t[:3]
		return cls(mtx)

	@property
	def R(self):
		return self.M[:3, :3]

	@property
	def t(self):
		return self.M[:3, 3]

	def __matmul__(self, other):
		return Transform(self.M @ other.M)

	def inverse(self):
		return Transform.from_Rt(self.R.T, -self.R.T @ self.t)

	def apply_point(self, p=None):
		p = np.zeros(3) if p is None else np.asarray(p)
		if p.ndim == 1: return (self.M @ np.append(p, 1.0))[:3]
		elif p.ndim == 2:
			points = np.hstack([p, np.ones((p.shape[0], 1))])
			return ((self.M @ points.T).T)[:, :3]
		raise NotImplementedError()

	def apply_vector(self, v=None):
		v = np.zeros(3) if v is None else np.asarray(v)
		if v.ndim == 1: return (self.M @ np.append(v, 0.0))[:3]
		elif  v.ndim == 2:
			vecs = np.hstack([v, np.zeros((v.shape[0], 1))])
			return ((self.M @ vecs.T).T)[:, :3]
		raise NotImplementedError()

	def __repr__(self):
		return 'Transform(\n%s)' % np.array2string(self.M, precision=4, suppress_small=True)

#region TASK 1

def task_1():
	T = Transform.from_Rt(rotz(np.pi / 6) @ rotx(np.pi / 4), [1.0, -2.0, 0.5])
	resid = np.linalg.norm((T @ T.inverse()).M - np.eye(4))
	print('||T @ T^-1 - I|| = %.3e' % resid)
	assert resid < 1e-12, 'обращение реализовано неверно'
	print('OK')

#endregion

#region TASK 2

def task_2():
	T = Transform.from_Rt(R=None, t = (1,2,3))
	shift_1 = [1,2,3]
	shift_2 = [[9,8,7], [8,7,6], [7,6,5]]
	print(f"apply_point: {T.apply_point(shift_1)}")		# apply_point:  [2. 4. 6. 1.]
	print(f"apply_point:\n{T.apply_point(shift_2)}")	# apply_point:  [[10. 10. 10.], [9. 9. 9.], [8. 8. 8.]]
	print(f"apply_vector: {T.apply_vector(shift_1)}")	# apply_vector:  [1. 2. 3. 0.]
	print(f"apply_vector:\n{T.apply_vector(shift_2)}")	# apply_vector:  [[9. 8. 7.], [8. 7. 6.], [7. 6. 5.]]

#endregion

#region TASK 3

def random_transform(rng):
	a = rng.uniform(-np.pi, np.pi, size=3)
	R = rotx(a[0]) @ roty(a[1]) @ rotz(a[2])
	t = rng.uniform(SHIFT_MIN, SHIFT_MAX, size=3)
	return Transform.from_Rt(R, t)

def task_3():
	A = random_transform(rng)
	B = random_transform(rng)
	C = random_transform(rng)

	assoc = np.linalg.norm(((A @ B) @ C).M - (A @ (B @ C)).M)
	print("Ассоциативность\t\t||(AB)C - A(BC)||\t= %.3e" % assoc)

	necom = np.linalg.norm((A @ B).M - (B @ A).M)
	print("некоммутативность\t||AB - BA||\t\t= %.3f" % necom)

	R = (A @ B).R
	a = np.linalg.norm(R.T @ R - np.eye(3))
	det_R = np.linalg.det(R)
	print("||R^T R - I||\t= %.3e\ndet(R)\t\t= %.3f" % (a, det_R))	

#endregion

#region TASK 4

PARENT = {'base': 'world', 'table': 'world', 'shoulder': 'base', 'elbow': 'shoulder',
		  'wrist': 'elbow', 'camera': 'wrist', 'object': 'table'}

LOCAL = {
	'base':     Transform.from_Rt(rotz(0.30), [0.00, 0.00, 0.20]),
	'table':    Transform.from_Rt(np.eye(3),  [1.20, 0.40, 0.00]),
	'shoulder': Transform.from_Rt(roty(0.45), [0.00, 0.00, 0.35]),
	'elbow':    Transform.from_Rt(roty(-0.80), [0.40, 0.00, 0.00]),
	'wrist':    Transform.from_Rt(rotx(0.60), [0.35, 0.00, 0.00]),
	'camera':   Transform.from_Rt(rotz(-np.pi / 2) @ roty(np.pi / 2), [0.05, 0.00, 0.10]),
	'object':   Transform.from_Rt(rotz(1.10), [0.10, -0.15, 0.75]),
}

def chain_to_root(frame):
	chain = [frame]
	while frame in PARENT:
		frame = PARENT[frame]
		chain += [frame]
	return chain

def world_from(frame):
	if frame not in PARENT:
		return Transform()
	return world_from(PARENT[frame]) @ LOCAL[frame]

def lookup(target, source):
	return world_from(target).inverse() @ world_from(source)

def task_4():
	cam_obj = lookup('camera', 'object').apply_point()
	print('Положение object в СК camera',  cam_obj)

#endregion

#region TASK 5

def task_5():
	axes_length = 0.05
	axis_colors = ['r', 'g', 'b']   # x, y, z

	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='3d')

	frames = list(LOCAL.keys()) + ['world']
	for frame in frames:
		Twf = world_from(frame)
		origin = Twf.apply_point([0.0, 0.0, 0.0])
		axes_local = np.eye(3) * axes_length 
		for i in range(3):
			tip = Twf.apply_point(axes_local[i])
			ax.plot(*zip(origin, tip), color=axis_colors[i], linewidth=2)
		ax.text(*origin, frame, fontsize=9)

	ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
	ax.set_box_aspect([1, 1, 1])
	plt.title('Дерево систем координат')
	plt.show()

#endregion

#region TASK 6

# def random_transform(rng, rot_scale=1e-3, t_scale=1e-3):
#     angles = rng.normal(scale=rot_scale, size=3)
#     R = rotz(angles[0]) @ roty(angles[1]) @ rotx(angles[2])
#     t = rng.normal(scale=t_scale, size=3)
#     return Transform.from_Rt(R, t)

def task_6():
	Tstep = random_transform(rng)
	Tstep_inv = Tstep.inverse()

	Ns = np.unique(np.logspace(1, 5, 25).astype(int))
	err_T = []      # ||T_acc - I||
	err_R = []      # ||R^T R - I||

	for N in Ns:
		Tacc = Transform()               # тождественное
		for _ in range(N):
			Tacc = Tacc @ Tstep
		for _ in range(N):
			Tacc = Tacc @ Tstep_inv
		err_T.append(np.linalg.norm(Tacc.M - np.eye(4)))
		err_R.append(np.linalg.norm(Tacc.R.T @ Tacc.R - np.eye(3)))

	err_T = np.array(err_T)
	err_R = np.array(err_R)

	# Оценка наклона в дважды логарифмическом масштабе
	slope_T, intercept_T = np.polyfit(np.log(Ns), np.log(err_T), 1)
	slope_R, intercept_R = np.polyfit(np.log(Ns), np.log(err_R), 1)
	print('наклон ||T_acc - I|| ~ N^%.3f' % slope_T)
	print('наклон ||R^TR - I|| ~ N^%.3f' % slope_R)

	fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
	axs[0].loglog(Ns, err_T, 'o-')
	axs[0].set_title(r'$\|T_{acc}-I\|$'); axs[0].set_xlabel('N'); axs[0].grid(True, which='both')
	axs[1].loglog(Ns, err_R, 'o-')
	axs[1].set_title(r'$\|R^TR-I\|$'); axs[1].set_xlabel('N'); axs[1].grid(True, which='both')
	plt.tight_layout(); plt.show()

#endregion

#region TASK 7

def orthogonalize(R):
    U, S, Vt = np.linalg.svd(R)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt

def orth_error(R):
    return np.linalg.norm(R.T @ R - np.eye(3))

def task_7():
	R_init = rotz(0.7) @ roty(-0.4) @ rotx(0.2)

	sigma = 2e-3
	R_noisy = R_init + rng.normal(scale=sigma, size=(3, 3))
	R_fixed = orthogonalize(R_noisy)

	print('до исправления:')
	print('  ошибка относительно R_init:\t', np.linalg.norm(R_noisy - R_init))
	print('  неортогональность:\t\t', orth_error(R_noisy))
	print('после исправления:')
	print('  ошибка относительно R_init:\t', np.linalg.norm(R_fixed - R_init))
	print('  неортогональность:\t\t', orth_error(R_fixed))

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