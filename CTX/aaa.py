#coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。
import sys
import numpy as np


if __name__ == "__main__":
    mn = sys.stdin.readline().strip()
    print(mn)
    if mn[1] == " ":
        m = int(mn[0])
    else:
        m = int(mn[0:2])

    if mn[1] == " ":
        n = int(mn[2:])
    else:
        n = int(mn[3:])

    print("m: ", m)
    print("n: ", n)

    if (m <= 1 or m >= 50) or (n <= 1 or n >= 50):
        print("error")
    else:
        values = []
        numbers = []
        for i in range(m):
            line = sys.stdin.readline().strip()
            value_each_line = list(map(int, line.split()))
            values.append(value_each_line)
        for i in range(m):
            for j in range(n):
                v = values[i][j]
                print(v)
                numbers.append(v)
        numbers = np.sort(numbers)
        print(numbers)
        for i in range(m):
            for j in range(n):
                if j != n-1:
                    print(numbers[i * m + j], " ", end="")
                else:
                    print(numbers[i * m + j], "\n", end="")


        def next_value(arr, i, j):
            if i != len(arr) - 1 and j == len(arr[0]) - 1:  # 向下排列
                i = i + 1
                return [i, j]
            if i == len(arr) - 1 and j != 0:  # 向左排列
                j = j - 1
                return [i, j]
            if i == 0 and j != len(arr[0]) - 1:  # 向右排列
                j = j + 1
                return [i, j]
            if i != 0 and j == 0:  # 向上排列
                i = i - 1
                if arr[i][j] == " ":
                    i = i + 1
                    j = j + 1
                    return [i, j]

            elif i != 0 and j != 0:
                if arr[i - 1][j] == " " and arr[i][j + 1] != " ":
                    j = j + 1
                    return [i, j]
                if arr[i][j + 1] == " " and arr[i + 1][j] != " ":
                    i = i + 1
                if arr[i + 1][j] == " " and arr[i][j - 1] != " ":
                    j = j - 1
                    return [i, j]
                if arr[i][j - 1] == " " and arr[i - 1][j] != " ":
                    i = i - 1
                    return [i, j]
            else:
                print("over")

        print(values)
        i = 0
        j = 0
        for n in range((len(numbers))):
            temp = next_value(values, i, j)
            values[i][j] = " "
            if n == len(numbers) - 1: break
            i = temp[0]
            j = temp[1]
        print(values)



