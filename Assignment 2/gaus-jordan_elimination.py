A = [[1, 0, 2],
     [2, -1, 3],
     [4, 1, 8]]

b = [[1],
     [-1],
     [2]]

# Create Augmented Matrix
C = [[1, 0, 2, 1],
     [2, -1, 3, -1],
     [4, 1, 8, 2]]

# Set E = 1
E = 1

for j in range(len(C[0])):
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


if E == 1:
    print("A unique solution was found!")
    print("x =", C[0][-1])
    print("y =", C[1][-1])
    print("z =", C[2][-1])
else:
    print("The algorithm failed to find a unique solution!")
