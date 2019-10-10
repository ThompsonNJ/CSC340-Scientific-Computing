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

        # For all rows i > j: row i = row i - C[i][j]/C[j][j] * row j
        for row in range(len(C)):
            if row > j:
                numerator = C[row][j]
                denominator = C[j][j]
                for col in range(len(C[j])):
                    C[row][col] -= numerator / denominator * C[j][col]

if E == 1:
    print("A unique solution was found!")

    # Partition matrix C into C = [D | e]
    D = [[0 for col in range(len(A[0]))] for row in range(len(A))]
    for row in range(len(A)):
        for col in range(len(A[0])):
            D[row][col] = C[row][col]

    e = [[0 for col in range(len(b[0]))] for row in range(len(b))]
    for row in range(len(b)):
        for col in range(len(b[0])):
            e[row][col] = C[row][-1]

    # Create a matrix of n zeros and uses substitution to extract the solution
    x = [0 for i in range(len(D))]
    for j in range(len(D) - 1, -1, -1):
        summation = 0
        for i in range(j + 1, len(D)):
            summation += D[j][i] * x[i]

        x[j] = (1 / D[j][j]) * (e[j][0] - summation)

    print("x =", x[0])
    print("y =", x[1])
    print("z =", x[2])

else:
    print("The algorithm failed to find a unique solution!")
