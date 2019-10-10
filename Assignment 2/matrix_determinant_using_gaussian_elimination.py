A = [[1, -1, 0],
     [-2, 2, -1],
     [0, 1, -2]]

# Set r = 0
r = 0
# Set E = 1
E = 1

for j in range(len(A[0])):
    # Find the largest magnitude value in column j only looking at row j and below.
    if j < len(A):
        mag = A[j][j]
        p = j
        for i in range(j, len(A)):
            if abs(A[i][j]) > abs(mag):
                mag = A[i][j]
                p = i

        # If A[p][j] = 0: set E = 0, exit
        if A[p][j] == 0:
            E = 0
            break

        # If p > j: flip rows p and j
        if p > j:
            r += 1
            temp = A[p]
            A[p] = A[j]
            A[j] = temp

        # For all rows i > j: row i = row i - A[i][j]/C[j][j] * row j
        for row in range(len(A)):
            if row > j:
                numerator = A[row][j]
                denominator = A[j][j]
                for col in range(len(A[j])):
                    A[row][col] -= numerator / denominator * A[j][col]

if E == 1:
    Adet_matrix = [[0 for col in range(len(A[0]))] for row in range(len(A))]
    for row in range(len(A)):
        for col in range(len(A[0])):
            Adet_matrix[row][col] = A[row][col]

    detA = (-1) ** r
    for row in range(len(Adet_matrix)):
        detA *= Adet_matrix[row][row]

    print("A unique solution was found!")
    print("The determinant of A =", detA)


else:
    print("The algorithm failed to find a unique solution!")
    print("The determinant of A = 0, A is singular!")
