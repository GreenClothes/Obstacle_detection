import numpy as np

a = [1, 2, 3]
b = [4, 5, 6]

def ab(a, b):
    m = int(input().strip())
    if m < 5:
        a[0] = b[0]
    else:
        a[0] = b[1]

while 1:
    ab(a, b)
    print(a)