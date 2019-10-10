import math

import cv2
import numpy as np


def center_image(original_image, original_rows, original_columns):
    window_size = int(max(original_rows, original_columns) * 3 / 2)
    new_image = np.zeros((window_size, window_size, 3), np.float32)
    new_rows = len(new_image)
    new_columns = len(new_image[0])
    translation_matrix = np.float32([[1, 0, (new_columns / 2) - (original_columns / 2)],
                                     [0, 1, (new_rows / 2) - (original_rows / 2)]])
    centered_matrix = np.zeros((new_rows, new_columns, 3), np.float32)

    for i in range(original_columns):
        for j in range(original_rows):
            for k in range(3):
                new_image[i][j][k] = original_image[i][j][k]

    for i in range(original_columns):
        for j in range(original_rows):
            xy_matrix = [[i],
                         [j],
                         [1]]

            new_xy = multiply_matrices(translation_matrix, xy_matrix)
            new_x_coord = int(round(new_xy[0][0]))
            new_y_coord = int(round(new_xy[1][0]))
            for k in range(3):
                centered_matrix[new_x_coord][new_y_coord][k] = original_image[i][j][k]

    return centered_matrix


def multiply_matrices(n, m):
    n_rows = len(n)
    n_columns = len(n[0])
    m_rows = len(m)
    m_columns = len(m[0])
    if n_columns == m_rows:
        output_matrix = np.zeros((n_rows, m_columns), np.float32)
        for i in range(n_rows):
            for j in range(m_columns):
                for k in range(n_columns):
                    output_matrix[i][j] += n[i][k] * m[k][j]

        return output_matrix

    raise ValueError


def rotate_image(rotation_angle, larger_image):
    rotation_matrix = [
        [math.cos(math.radians(rotation_angle)), -math.sin(math.radians(rotation_angle))],
        [math.sin(math.radians(rotation_angle)), math.cos(math.radians(rotation_angle))]
    ]
    rows = len(larger_image)
    columns = len(larger_image[0])
    rotated_matrix = np.zeros((rows, columns, 3), np.float32)
    displacement_error = 0

    for i in range(columns):
        for j in range(rows):
            xy_matrix = [[i - columns // 2],
                         [j - rows // 2]]

            new_xy = multiply_matrices(rotation_matrix, xy_matrix)

            new_x_coord = int(new_xy[0][0] + columns // 2)
            new_y_coord = int(new_xy[1][0] + rows // 2)
            displacement_error += pixel_displacement_error(i, j, new_x_coord, new_y_coord)
            if (new_x_coord > 0 and new_x_coord < columns) and (new_y_coord > 0 and new_y_coord < rows):
                for k in range(3):
                    rotated_matrix[i][j][k] = larger_image[new_x_coord][new_y_coord][k]

    return rotated_matrix, displacement_error


def pixel_displacement_error(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)


def scale_back_down(large_image, original_image):
    large_rows = len(large_image)
    large_columns = len(large_image[0])
    original_rows = len(original_image)
    original_columns = len(original_image[0])
    shifted_matrix = np.zeros((large_rows, large_columns, 3), np.float32)
    scaled_matrix = np.zeros((original_rows, original_columns, 3), np.float32)
    translation_matrix = np.float32([[1, 0, (large_columns / 2) - (original_columns / 2)],
                                     [0, 1, (large_rows / 2) - (original_rows / 2)]])

    for i in range(large_columns):
        for j in range(large_rows):
            xy_matrix = [[i],
                         [j],
                         [1]]

            new_xy = multiply_matrices(translation_matrix, xy_matrix)

            new_x_coord = int(new_xy[0][0])
            new_y_coord = int(new_xy[1][0])
            if (new_x_coord > 0 and new_x_coord < large_columns) and (new_y_coord > 0 and new_y_coord < large_rows):
                for k in range(3):
                    shifted_matrix[i][j][k] = large_image[new_x_coord][new_y_coord][k]

    for i in range(original_columns):
        for j in range(original_rows):
            for k in range(3):
                scaled_matrix[i][j][k] = shifted_matrix[i][j][k]

    return scaled_matrix


def absolute_color_error(original_image, scaled_rotated_image):
    original_image_rows = len(original_image)
    original_image_columns = len(original_image[0])
    color_error = 0
    num_pixels = original_image_rows * original_image_columns

    for i in range(original_image_columns):
        for j in range(original_image_rows):
            for k in range(3):
                color_error += abs(scaled_rotated_image[i][j][k] - original_image[i][j][k])

    return color_error / (3 * num_pixels)


def show_image(image, image_caption):
    return cv2.imshow(image_caption, image / 255.0)


original_image = cv2.imread("dogs.jpg")
original_image_height = original_image.shape[0]
original_image_width = original_image.shape[1]

centered_image = center_image(original_image, original_image_height, original_image_width)
rotation_angle = int(input("rotation angle?: "))
previous = centered_image

if rotation_angle > 0:
    num_iterations = 360 // rotation_angle
else:
    num_iterations = 1
count = 1
tot_displacement = 0
while count <= num_iterations:
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    rotated_image, rot_displacement = rotate_image(rotation_angle, previous)
    tot_displacement += rot_displacement
    rotation_so_far = str(rotation_angle * count)
    image_caption = "Image rotated " + rotation_so_far + " degrees with steps of " + str(rotation_angle) + "degrees " + str(
        count) + "rotations applied so far."
    #show_image(rotated_image, image_caption)
    previous = rotated_image
    count += 1

num_pixels = len(rotated_image) * len(rotated_image[0])
scaled_rotated_matrix = scale_back_down(rotated_image, original_image)
pixel_displacement = tot_displacement / (num_pixels * num_iterations)
print("Absolute Color Error:", absolute_color_error(original_image, scaled_rotated_matrix))
print("Pixel Displacement Error", pixel_displacement)
print("(# Rotations) * (Pixel Displacement)", num_iterations * pixel_displacement)


############## CLOSE FILE
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("savedImage.jpg", rotated_image)
