import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

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
		return Transform.from_Rt(np.transpose(self.R), -np.dot(np.transpose(self.R), self.t))

	def apply_point(self, p):
		p = np.zeros(3) if p is None else np.asarray(p)
		if p.ndim == 1: return (self.M @ np.append(p, 1.0))[:3]
		elif p.ndim == 2:
			points = np.hstack([p, np.ones((p.shape[0], 1))])
			return ((self.M @ points.T).T)[:, :3]
		raise NotImplementedError()

	def apply_vector(self, v):
		v = np.zeros(3) if v is None else np.asarray(v)
		if v.ndim == 1: return (self.M @ np.append(v, 0.0))[:3]
		elif  v.ndim == 2:
			vecs = np.hstack([v, np.zeros((v.shape[0], 1))])
			return ((self.M @ vecs.T).T)[:, :3]
		raise NotImplementedError()

	def __repr__(self):
		return 'Transform(\n%s)' % np.array2string(self.M, precision=4, suppress_small=True)


def task_1():
	T = Transform.from_Rt(rotz(np.pi / 6) @ rotx(np.pi / 4), [1.0, -2.0, 0.5])
	resid = np.linalg.norm((T @ T.inverse()).M - np.eye(4))
	print('||T @ T^-1 - I|| = %.3e' % resid)
	assert resid < 1e-12, 'обращение реализовано неверно'
	print('OK')

def task_2():
	T = Transform.from_Rt(R=None, t = (1,2,3))
	shift_1 = [1,2,3]
	shift_2 = [[9,8,7], [8,7,6], [7,6,5]]
	print(f"apply_point: {T.apply_point(shift_1)}")		# apply_point:  [2. 4. 6. 1.]
	print(f"apply_point:\n{T.apply_point(shift_2)}")	# apply_point:  [[10. 10. 10.], [9. 9. 9.], [8. 8. 8.]]
	print(f"apply_vector: {T.apply_vector(shift_1)}")	# apply_vector:  [1. 2. 3. 0.]
	print(f"apply_vector:\n{T.apply_vector(shift_2)}")	# apply_vector:  [[9. 8. 7.], [8. 7. 6.], [7. 6. 5.]]

task_2()