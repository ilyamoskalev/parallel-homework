import math


def get_divisors(n, flag):
    divisors = []
    i = 1
    while i <= math.sqrt(n):
        if n % i == 0:
            divisors.append(i)
            if n != 1:
                divisors.append(int(n / i))
            if flag is True:
                if n != 1:
                    divisors.append(int(-n / i))
                divisors.append(-i)
        i = i + 1

    return divisors
