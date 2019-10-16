from copy import deepcopy

import cv2
import numpy as np


def makeIMatrices(image):
    rows = len(image)
    columns = len(image[0])
    Ixx = np.zeros((rows, columns), np.float32)
    Iyy = np.zeros((rows, columns), np.float32)
    Ixy = np.zeros((rows, columns), np.float32)
    gradImX = cv2.Sobel(grayIm, cv2.CV_64F, 1, 0, ksize=3)
    gradImY = cv2.Sobel(grayIm, cv2.CV_64F, 0, 1, ksize=3)

    for i in range(rows):
        for j in range(columns):
            Ixx[i][j] = gradImX[i][j] * gradImX[i][j]
            Iyy[i][j] = gradImY[i][j] * gradImY[i][j]
            Ixy[i][j] = gradImX[i][j] * gradImY[i][j]

    return Ixx, Iyy, Ixy


def calcCornerness(Ixx, Iyy, Ixy):
    rows = len(Ixx)
    columns = len(Ixx[0])
    cornernessMatrix = np.zeros((rows, columns), np.float32)
    minCornerness = 0
    maxCornerness = 0
    for i in range(rows):
        for j in range(columns):
            sumIxx = 0
            sumIyy = 0
            sumIxy = 0
            for k in range(-1, 2):
                for L in range(-1, 2):
                    try:
                        if (i + k) < 0 or (j + L) < 0:
                            raise IndexError

                        sumIxx += Ixx[i + k][j + L]
                        sumIyy += Iyy[i + k][j + L]
                        sumIxy += Ixy[i + k][j + L]
                    except IndexError:
                        continue

            det = (sumIxx * sumIyy) - (sumIxy ** 2)
            trace = sumIxx + sumIxy
            cornerness = det - (0.05 * (trace))
            cornernessMatrix[i][j] = cornerness

            if cornerness > maxCornerness:
                maxCornerness = cornerness

            if cornerness < minCornerness:
                minCornerness = cornerness

    return cornernessMatrix, minCornerness, maxCornerness


def drawCornersTopPercentile(image, cornernessMatrix, minCornerness, maxCornerness):
    rows = len(cornernessMatrix)
    columns = len(cornernessMatrix[0])
    percent = .2
    threshold = percent * (maxCornerness - minCornerness) + minCornerness
    for i in range(rows):
        for j in range(columns):
            if cornernessMatrix[i][j] >= threshold:
                cv2.circle(image, (j, i), 5, (0, 0, 255), -1)

    return image


def drawCornersTopNumber(image, cornernessMatrix, minCornerness):
    rows = len(cornernessMatrix)
    columns = len(cornernessMatrix[0])
    num = 2000
    count = 0
    withIndices = []
    minWithIndices = minCornerness
    for i in range(rows):
        for j in range(columns):
            if count < num:
                withIndices.append([cornernessMatrix[i][j], i, j])
                count += 1
                if count == num:
                    minWithIndices = min(withIndices)
            else:
                if cornernessMatrix[i][j] > minWithIndices[0]:
                    minPos = withIndices.index(min(withIndices))
                    withIndices[minPos] = [cornernessMatrix[i][j], i, j]
                    minWithIndices = min(withIndices)

    for i in range(len(withIndices)):
        cv2.circle(image, (withIndices[i][2], withIndices[i][1]), 5, (0, 0, 255), -1)

    return image


def drawCornersTopNumberNeighborhoods(image, cornernessMatrix, minCornerness):
    rows = len(cornernessMatrix)
    columns = len(cornernessMatrix[0])
    num = 100
    neighborhoodSize = 100
    withIndices = []
    minWithIndices = minCornerness

    row = 0
    column = 0
    count = 0
    while row <= rows:
        while column <= columns:
            for i in range(neighborhoodSize):
                for j in range(neighborhoodSize):
                    try:
                        if count < num:
                            withIndices.append([cornernessMatrix[i + row][j + column], i + row, j + column])
                            count += 1
                            if count == num:
                                minWithIndices = min(withIndices)
                        else:
                            if cornernessMatrix[i + row][j + column] > minWithIndices[0]:
                                minPos = withIndices.index(min(withIndices))
                                withIndices[minPos] = [cornernessMatrix[i + row][j + column], i + row, j + column]
                                minWithIndices = min(withIndices)
                    except IndexError:
                        pass

            for i in range(len(withIndices)):
                cv2.circle(image, (withIndices[i][2], withIndices[i][1]), 5, (0, 0, 255), -1)

            del withIndices
            withIndices = []
            count = 0
            column += neighborhoodSize

        del withIndices
        withIndices = []
        count = 0
        column = 0
        row += neighborhoodSize

    return image


def drawCornersGrayscale(image, cornernessMatrix, minCornerness, maxCornerness):
    rows = len(cornernessMatrix)
    columns = len(cornernessMatrix[0])
    for i in range(rows):
        for j in range(columns):
            image[i][j] = (cornernessMatrix[i][j] - minCornerness) / (maxCornerness - minCornerness) * 255.0

    return image


def calcQuartiles(myList):
    middle = len(myList) // 2
    leftMid = len(myList) // 4
    rightMid = (len(myList) // 4) * 3
    if len(myList) % 2 == 1:
        median = myList[middle][0]
        firstQuartile = (myList[leftMid][0] + myList[leftMid - 1][0]) / 2
        thirdQuartile = (myList[rightMid][0] + myList[rightMid - 1][0]) / 2
    else:
        median = (myList[middle][0] + myList[middle - 1][0]) / 2
        firstQuartile = myList[leftMid][0]
        thirdQuartile = myList[rightMid][0]

    firstQuartileList = []
    secondQuartileList = []
    thirdQuartileList = []
    fourthQuartileList = []
    for i in range(len(myList)):
        if i <= leftMid:
            firstQuartileList.append(myList[i][0])
        elif i > leftMid and i <= middle:
            secondQuartileList.append(myList[i][0])
        elif i > middle and i <= rightMid:
            thirdQuartileList.append(myList[i][0])
        else:
            fourthQuartileList.append(myList[i][0])

    firstQuartileMin = min(firstQuartileList)
    firstQuartileMax = max(firstQuartileList)

    secondQuartileMin = min(secondQuartileList)
    secondQuartileMax = max(secondQuartileList)

    thirdQuartileMin = min(thirdQuartileList)
    thirdQuartileMax = max(thirdQuartileList)

    fourthQuartileMin = min(fourthQuartileList)
    fourthQuartileMax = max(fourthQuartileList)

    return firstQuartile, firstQuartileMin, firstQuartileMax, secondQuartileMin, secondQuartileMax, median, thirdQuartile, thirdQuartileMin, thirdQuartileMax, fourthQuartileMin, fourthQuartileMax


def drawCornersColored(image, cornernessMatrix, minCornerness, maxCornerness):
    rows = len(cornernessMatrix)
    columns = len(cornernessMatrix[0])
    withIndices = []
    for i in range(rows):
        for j in range(columns):
            withIndices.append([cornernessMatrix[i][j], i, j])

    withIndices.sort()
    firstQuartile, firstQuartileMin, firstQuartileMax, secondQuartileMin, secondQuartileMax, median, thirdQuartile, thirdQuartileMin, thirdQuartileMax, fourthQuartileMin, fourthQuartileMax = calcQuartiles(
        withIndices)
    for i in range(len(withIndices)):
        # green to yellow
        if withIndices[i][0] < firstQuartile:
            image[withIndices[i][1]][withIndices[i][2]][0] = 0
            image[withIndices[i][1]][withIndices[i][2]][1] = (cornernessMatrix[withIndices[i][1]][
                                                                  withIndices[i][2]] - firstQuartileMin) / (
                                                                     firstQuartileMax - firstQuartileMin) * 255.0
            image[withIndices[i][1]][withIndices[i][2]][2] = (1 - (cornernessMatrix[withIndices[i][1]][
                                                                      withIndices[i][2]] - firstQuartileMin) / (
                                                                     firstQuartileMax - firstQuartileMin)) * 255.0
        # yellow to orange
        elif withIndices[i][0] >= firstQuartile and withIndices[i][0] < median:
            image[withIndices[i][1]][withIndices[i][2]][0] = 0
            image[withIndices[i][1]][withIndices[i][2]][1] = (1 - (cornernessMatrix[withIndices[i][1]][
                                                                      withIndices[i][2]] - secondQuartileMin) / (
                                                                     secondQuartileMax - secondQuartileMin)) * 165.0
            image[withIndices[i][1]][withIndices[i][2]][2] = (cornernessMatrix[withIndices[i][1]][
                                                                  withIndices[i][2]] - secondQuartileMin) / (
                                                                     secondQuartileMax - secondQuartileMin) * 255.0
        # orange to red
        elif withIndices[i][0] >= median and withIndices[i][0] < thirdQuartile:
            image[withIndices[i][1]][withIndices[i][2]][0] = 0
            image[withIndices[i][1]][withIndices[i][2]][1] = (cornernessMatrix[withIndices[i][1]][
                                                                  withIndices[i][2]] - thirdQuartileMin) / (
                                                                     thirdQuartileMax - thirdQuartileMin) * 165.0
            image[withIndices[i][1]][withIndices[i][2]][2] = (1 - (cornernessMatrix[withIndices[i][1]][
                                                                      withIndices[i][2]] - thirdQuartileMin) / (
                                                                     thirdQuartileMax - thirdQuartileMin)) * 255.0
        # red to fuchsia
        elif withIndices[i][0] >= thirdQuartile:
            image[withIndices[i][1]][withIndices[i][2]][0] = (cornernessMatrix[withIndices[i][1]][
                                                                      withIndices[i][2]] - fourthQuartileMin) / (
                                                                     fourthQuartileMax - fourthQuartileMin) * 255.0
            image[withIndices[i][1]][withIndices[i][2]][1] = (1 - (cornernessMatrix[withIndices[i][1]][
                                                                      withIndices[i][2]] - fourthQuartileMin) / (
                                                                     fourthQuartileMax - fourthQuartileMin)) * 255.0
            image[withIndices[i][1]][withIndices[i][2]][2] = (cornernessMatrix[withIndices[i][1]][
                                                                  withIndices[i][2]] - fourthQuartileMin) / (
                                                                     fourthQuartileMax - fourthQuartileMin) * 255.0

    return image


image = cv2.imread("tunnel.jpg")
grayIm = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

Ixx, Iyy, Ixy = makeIMatrices(grayIm)
cornernessMatrix, minCornerness, maxCornerness = calcCornerness(Ixx, Iyy, Ixy)

# topPercentile_image_copy = deepcopy(image)
# topPercentile = drawCornersTopPercentile(topPercentile_image_copy, cornernessMatrix, minCornerness, maxCornerness)
# cv2.imshow("topPercentile", topPercentile)
#
# topNumber_image_copy = deepcopy(image)
# topNumber = drawCornersTopNumber(topNumber_image_copy, cornernessMatrix, minCornerness)
# cv2.imshow("topNumber", topNumber)
#
# topNumberNeighborhoods_image_copy = deepcopy(image)
# topNumberNeighborhoods = drawCornersTopNumberNeighborhoods(topNumberNeighborhoods_image_copy, cornernessMatrix,
#                                                            minCornerness)
# cv2.imshow("topNumberNeighborhoods", topNumberNeighborhoods)
#
grayBonus_copy = deepcopy(image)
grayBonus = drawCornersGrayscale(grayBonus_copy, cornernessMatrix, minCornerness, maxCornerness)
cv2.imwrite("grayscale.png", grayBonus)

colored_copy = deepcopy(image)
cornersColored = drawCornersColored(colored_copy, cornernessMatrix, minCornerness, maxCornerness)
cv2.imwrite("rgb.png", cornersColored)

cv2.waitKey(0)
cv2.destroyAllWindows()
