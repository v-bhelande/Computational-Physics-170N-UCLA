import numpy as np
from typing import List

# 1.1 Function Definition
def recursive_factorial(n: int):
    if n < 0:
        return "n must be a non-negative integer! Exiting function..."
    elif int(n) != n:
        return "n must be an integer! Exiting function..."
    elif n == 0 or n == 1:
        return 1
    else:
        return n*recursive_factorial(n-1)

# 1.2a Function Definition
fib_memo = {}

def F_memo(n):
    if n == 1:
        return 0
    elif n == 2:
        return 1
    elif n not in fib_memo:
        fib_memo[n] = F_memo(n - 1) + F_memo(n - 2)
    return fib_memo[n]

# 1.2b Function Definition
def recursive_factorial_memo(n):
    if n == 1:
        return 0
    elif n == 2:
        return 1
    elif n not in fib_memo:
        fib_memo[n] = F_memo(n - 1) + F_memo(n - 2)
    return fib_memo[n]

# 1.3a Function Definition
def binomial_factorial(n: int, k: int) -> int:
    n_fact = recursive_factorial(n)
    k_fact = recursive_factorial(k)
    n_k_fact = recursive_factorial(n-k)
    if isinstance(n_fact, int) and isinstance(k_fact, int) and isinstance(n_k_fact, int):
        return int(n_fact/(k_fact*n_k_fact))
    else:
        return "Invalid number entered, exiting function..."
    
# 1.3b Function Definition
def binomial_recursive(n: int, k: int) -> int:
    if int(n) != n or int(k) != k:
        return "n and k must be integers! Exiting function..."
    if n < 0 or k < 0:
        return "n and k must be non-negative integers! Exiting function..."
    # Base Cases
    if k == 0 or k == n:
        return 1
    # Recursive function call
    else:
        return (binomial_recursive(n-1, k-1) + binomial_recursive(n-1, k))
    
# 1.4 Function Definition
log_dict = {}
def logistic(n: int, r: float, x0: float):
    if n < 0 or int(n) != n:
        return "n must be a non-negative integer! Exiting function..."
    # Base Case
    if n == 0:
        return r*x0*(1-x0)
    elif n not in log_dict:
        log_dict[n] = r*logistic(n-1, r, x0)*(1-logistic(n-1, r, x0))
    return log_dict[n]

# 2.1 Function Definition
from typing import List

def linear_search(L: List[int], n):
    # Check if any element in list is not a postive integer
    for i in range(len(L)):
        if int(L[i]) != L[i] or L[i] < 0:
            print("Element", i, " of list must be a postive integer! Exiting function...")
            return
    for i in range(len(L)):
        if (L[i] == n):
            return i
    return "The specific number is not in the list."

# 2.2 Function Definiton
def bisection_search(L: List[int], n):
    # Check if any element in list is not a postive integer
    for i in range(len(L)):
        if int(L[i]) != L[i] or L[i] < 0:
            print("Element", i, " of list must be a postive integer! Exiting function...")
            return
    # Check if list is ascending
    for i in range(1, len(L)):
        if L[i] < L[i-1]:
            print("List must be in ascending order! Exiting function...")
            return
    low = 0
    high = len(L)-1
    while low <= high:
        midpoint = low + (high - low)//2
        if (n == L[midpoint]):
            return midpoint
        elif (n > L[midpoint]):
            low = midpoint + 1
        else:                       
            high = midpoint - 1
    return "The specific number is not in the list."

# 2.3 Function Definiton
def bisection_root(f, x_left, x_right, epsilon):
    midpoint = (x_left+x_right)/2
    if np.abs(f(midpoint)) < epsilon:
        return midpoint
    elif np.sign(f(x_left)) == np.sign(f(midpoint)):
        return bisection_root(f, midpoint, x_right, epsilon)
    elif np.sign(f(x_right)) == np.sign(f(midpoint)):
        return bisection_root(f, x_left, midpoint, epsilon)
    else:
        return "Root was not found..."

# 2.4 Function Defintions
    
# 1. phi = 0
def phi_0(theta):
    v = 10 #m/s
    g = 9.81 #m/s
    return (2*v*np.cos(2*theta))/g

# 2. phi = pi/8
def phi_pi_8(theta):
    v = 10 #m/s
    g = 9.81 #m/s
    return (2*v*np.cos(2*theta+(np.pi/4)))/(g*np.cos(np.pi/8))

# 3. phi = pi/4
def phi_pi_4(theta):
    v = 10 #m/s
    g = 9.81 #m/s
    return (2*v*np.cos(2*theta+(np.pi/2)))/(g*np.cos(np.pi/4))

# 4. phi = 3*pi/8
def phi_3pi_8(theta):
    v = 10 #m/s
    g = 9.81 #m/s
    return (2*v*np.cos(2*theta+(3*np.pi/4)))/(g*np.cos(3*np.pi/8))

# 3.1 Function Definition
def sieve_Eratosthenes(n: int):
    if n < 2 or int(n) != n:
        return "n must be an integer greater than or equal to 2! Exiting function..."
    primes = [True for i in range(n+1)]
    primes[0] = False   # 0 is not a prime number
    primes[1] = False   # 1 is not a prime number
    # Start with first prime number and mark all its multiples as false in array
    prime_num = 2
    while (prime_num * prime_num <= n):
        if (primes[prime_num] == True):
            for i in range(prime_num*prime_num, n+1, prime_num):
                primes[i] = False
        prime_num += 1

    primes_list = []
    # Print prime numbers
    for prime_num in range(2, n+1):
        if (primes[prime_num] == True):
            primes_list.append(prime_num)
    return primes_list

# 3.2 Function Definition
def prime_factors(n):
    pf_list = []
    if n < 2 or int(n) != n:
        return "n must be an integer greater than or equal to 2! Exiting function..."
    # Divide n by 2 as much as possibel
    while n%2 == 0:
        pf_list.append(2)
        n = n//2
        if n == 1:
                return pf_list
    # Move on to other prime factors
    for i in range(3, n, 2):
        while n%i == 0:
            pf_list.append(i)
            n = n//i
            if n == 1:
                return pf_list
            
# 3.3 Function Definition
def nth_prime(n: int):
    if n <= 0 or int(n) != n:
        return "n must be a positive integer! Exiting function..."
    # Generate all prime numbers ahead of time
    primes_nums_list = sieve_Eratosthenes(100005)
    return primes_nums_list[n-1]

# 4 Function Definiton
def total_die(num_die: int, max_val: int, total_sum: int):
    if num_die == 0:
        return 1 if (total_sum == 0) else 0
    
    # If total sum can't be achieved
    if total_sum < 0 or num_die > total_sum or num_die*max_val < total_sum:
        return 0
    
    combos = 0
    for i in range(1, max_val+1):
        combos += total_die(num_die-1, max_val, total_sum-i)
    return combos