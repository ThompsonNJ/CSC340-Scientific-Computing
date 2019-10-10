import numpy as np

A = [[1, -1, 0],
     [-2, 2, -1],
     [0, 1, -2]]

# Construct augmented n x (2n) matrix from A and I.
C = [[1, -1, 0, 1, 0, 0],
     [-2, 2, -1, 0, 1, 0],
     [0, 1, -2, 0, 0, 1]]

# Set E = 1
E = 1

for j in range(len(A[0])):
    # Find the largest magnitude value in column j only looking at row j and below.
    if j < len(C):
        mag = C[j][j]
        p = j
        for i in range(j, len(C)):
            if abs(C[i][j]) > abs(mag):
                mag = C[i][j]
                p = i

        # If C[p][j] = 0: set E = 0, exit
        if C[p][j] == 0:
            E = 0
            break

        # If p > j: flip rows p and j
        if p > j:
            temp = C[p]
            C[p] = C[j]
            C[j] = temp

        # Divide row j by pivot value
        for col in range(len(C[j])):
            C[j][col] /= mag

        # For remaining rows i != j: row i = row i - C[i][j] * row j
        for row in range(len(C)):
            if row != j:
                mult = C[row][j]
                for col in range(len(C[j])):
                    C[row][col] -= mult * C[j][col]

# Create Ainverse using the right half of C
Ainverse = [[0 for col in range(len(A[0]))] for row in range(len(A))]
for row in range(len(C)):
    Ainverse[row] = C[row][len(C[0]) // 2:]

print("Matrix A:")
for row in A:
    print(row)

print("\nMatrix Ainverse:")
for row in Ainverse:
    print(row)

# Multiply A and Ainverse to check if I get the I back. Even though the assignment says using dot is prohibited, you
# said that this was okay since I am just using it to check my answer.
I = np.dot(A, Ainverse)
print("\nI:\n{}".format(I))
