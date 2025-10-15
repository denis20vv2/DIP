import numpy as np

n = 30

rnd = np.random.default_rng()

randomMatrix1 = rnd.integers(0, 50, 100)
randomMatrix2 = rnd.integers(0, 50, 100)

print()
print(randomMatrix1)
print()
print(randomMatrix2)
print()

sameValues = randomMatrix1 == randomMatrix2

indexes = np.where(sameValues)[0]
if indexes.size > 0:
    print("индексы:", indexes)
    print()
    print("значения:", randomMatrix1[indexes])
else:
    print("Совпадений не найдено:")