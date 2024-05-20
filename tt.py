import numpy as np
t1 = [[1, 1, 2], [1, 2, 3], [2, 3, 4]]
t2 = [[2, 1, 2], [1, 2, 3], [2, 3, 4]]
t3 = [[3, 1, 2], [1, 2, 3], [2, 3, 4]]
t4 = [[4, 1, 2], [1, 2, 3], [2, 3, 4]]

tt = [[]for i in range(10)]
print(tt)

tt[0] = tt[0]+t1
tt[0] = tt[0]+t4
tt[0] = np.reshape(tt[0], (2, 3, 3))
print(tt)
# tt[0].insert(0, t1)
# tt[1].insert(1, t2)
# tt[0].insert(2, t3)
# print(tt)
# print(tt[0])
# print(tt[0][0])
# print(tt[0][0][0])

# a = np.arange(12).reshape(3, 4)
# print(a)

# b = np.reshape(a, (3, 4, 1))
# print(b)
# np.shape(b)

# c=np.concatenate([b,b],2)
# np.shape(c)
