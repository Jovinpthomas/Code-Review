def ig(num):
    total = 0
    temp = num
    while temp > 0:
        digit = temp % 10
        total += digit ** 3
        temp //= 10
    return num == total
