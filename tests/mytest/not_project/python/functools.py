from functools import reduce

lis = [1, 3, 5, 6, 2]
print("列表元素的总和为: ", end="")
print(reduce(lambda x, y: x + y, lis))