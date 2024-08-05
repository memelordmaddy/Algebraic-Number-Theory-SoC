import random
from typing import List, Tuple, Union, Dict, Optional
def pair_gcd(a: int, b: int) -> int:
    """
    Compute the greatest common divisor (GCD) of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The GCD of a and b.
    """
    if b == 0:
        return a
    else:
        return pair_gcd(b, a % b)

def pair_egcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithm to find x, y, and gcd such that ax + by = gcd(a, b).

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        Tuple[int, int, int]: A tuple (x, y, gcd) where gcd is gcd(a, b) and ax + by = gcd.
    """
    if b == 0:
        return 1, 0, a
    else:
        x, y, gcd = pair_egcd(b, a % b)
        return y, x - y * (a // b), gcd

def are_relatively_prime(a: int, b: int) -> bool:
    """
    Check if two integers are relatively prime (i.e., their GCD is 1).

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        bool: True if a and b are relatively prime, False otherwise.
    """
    return pair_gcd(a, b) == 1

def gcd(*input: int) -> int:
    """
    Compute the greatest common divisor (GCD) of multiple integers.

    Args:
        *input (int): The integers for which the GCD is to be computed.

    Returns:
        int: The GCD of the input integers.
    """
    if len(input) == 2:
        return pair_gcd(input[0], input[1])
    else:
        op = input[0]
        for i in input[1:]:
            op = pair_gcd(op, i)
        return op

def pair_lcm(a: int, b: int) -> int:
    """
    Compute the least common multiple (LCM) of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The LCM of a and b.
    """
    return (a // pair_gcd(a, b)) * b

def lcm(*input: int) -> int:
    """
    Compute the least common multiple (LCM) of multiple integers.

    Args:
        *input (int): The integers for which the LCM is to be computed.

    Returns:
        int: The LCM of the input integers.
    """
    if len(input) == 2:
        return pair_lcm(input[0], input[1])
    else:
        op = input[0]
        for i in input[1:]:
            op = pair_lcm(op, i)
        return op

def mod_inv(a: int, n: int) -> int:
    """
    Compute the modular inverse of a modulo n.

    Args:
        a (int): The integer for which the modular inverse is to be computed.
        n (int): The modulus.

    Returns:
        int: The modular inverse of a mod n.

    Raises:
        Exception: If a and n are not coprime.
    """
    if not are_relatively_prime(a, n):
        raise Exception(f"{a} and {n} are not coprime")
    x, _, _ = pair_egcd(a, n)
    return x % n

def crt(a: List[int], n: List[int]) -> int:
    """
    Solve the Chinese Remainder Theorem for simultaneous congruences.

    Args:
        a (List[int]): The remainders for each congruence.
        n (List[int]): The moduli for each congruence.

    Returns:
        int: The solution to the simultaneous congruences.
    """
    pi = 1
    for modulus in n:
        pi *= modulus
    
    op = 0
    for i in range(len(n)):
        mi = n[i]
        ai = a[i]
        inv = mod_inv(pi // mi, mi)
        op += ((pi // mi) * inv * ai) % pi
    
    return op % pi

def pow(a: int, m: int, n: int) -> int:
    """
    Compute a to the power m modulo n.

    Args:
        a (int): The base.
        m (int): The exponent.
        n (int): The modulus.

    Returns:
        int: The result of (a^m) % n.
    """
    if m == 0:
        return 1
    if m < 0:
        return mod_inv(pow(a, -m, n), n)
    
    temp = pow(a, m // 2, n)
    temp = (temp * temp) % n
    if m % 2 == 0:
        return temp
    else:
        return (temp * a) % n

def is_quadratic_residue_prime(a: int, p: int) -> int:
    """
    Check if a is a quadratic residue modulo a prime p.

    Args:
        a (int): The integer to check.
        p (int): The prime modulus.

    Returns:
        int: 1 if a is a quadratic residue modulo p, -1 if it is not, and 0 if a and p are not coprime.
    """
    if p == 2:
        return 1
    if not are_relatively_prime(a, p):
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1

def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:
    """
    Check if a is a quadratic residue modulo p^e where p is a prime.

    Args:
        a (int): The integer to check.
        p (int): The prime base.
        e (int): The exponent.

    Returns:
        int: 1 if a is a quadratic residue modulo p^e, -1 if it is not, and 0 if a and p^e are not coprime.
    """
    if p == 1:
        return 1
    if not are_relatively_prime(a, p):
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def floor_sqrt(n: int) -> int:
    """
    Computes the floor of the square root of a given integer.

    Args:
        n (int): The integer value for which to compute the square root.

    Returns:
        int: The floor of the square root of `n`.
    """
    len_n = n.bit_length()
    k = (len_n - 1) // 2
    m = 2**k
    two_raised_to_i = 2**(k - 1)
    for i in range(k - 1, -1, -1):
        if (m + two_raised_to_i) ** 2 <= n:
            m = m + two_raised_to_i
        two_raised_to_i /= 2
    return int(m)

def floor_k_root(n: int, k: int) -> int:
    """
    Computes the floor of the k-th root of a given integer.

    Args:
        n (int): The integer value for which to compute the k-th root.
        k (int): The root to compute (e.g., 2 for square root, 3 for cube root).

    Returns:
        int: The floor of the k-th root of `n`.
    """
    len_n = n.bit_length()
    m = 2**((len_n - 1) // k)
    two_raised_to_i = 2**((len_n - 1) // (2 * k))
    for i in range((len_n - 1) // (2 * k), -1, -1):
        if (m + two_raised_to_i) ** k <= n:
            m = m + two_raised_to_i
        two_raised_to_i /= 2
    return int(m)

def is_perfect_power(n: int) -> bool:
    """
    Determines if a given integer is a perfect power.

    Args:
        n (int): The integer value to check.

    Returns:
        bool: True if `n` is a perfect power, False otherwise.
    """
    len_n = n.bit_length()
    for i in range(2, len_n):
        k = floor_k_root(n, i)
        if k ** i == n:
            return True
    return False

def natural_log(n: float) -> float:
    """
    Computes the natural logarithm of a given floating-point number.

    Args:
        n (float): The number to compute the natural logarithm for.

    Returns:
        float: The natural logarithm of `n`.
    """
    e = 2.718281828459045
    e_pow = e
    k = 1
    while e_pow * e <= n:
        e_pow *= e
        k += 1
    n /= e_pow
    n -= 1
    natural_log = 0
    n_cp = n
    for i in range(1, 10):
        if i % 2 == 0:
            natural_log -= n / i
        else:
            natural_log += n / i
        n *= n_cp
    return natural_log + k

def exp(n: float) -> float:
    """
    Computes the exponential function e^n, where e is the base of the natural logarithm.

    Args:
        n (float): The exponent to which e is raised.

    Returns:
        float: The value of e^n.
    """
    f = 1
    op = 1
    n_cp = n
    for i in range(1, 100):
        f *= i
        op += n / f
        n *= n_cp
    return op


class QuotientPolynomialRing:
    def __init__(self, poly: List[int], pi_gen: List[int]) -> None:
        """
        Initializes a QuotientPolynomialRing object with a polynomial and a quotienting polynomial.

        Args:
            poly (List[int]): The coefficients of the polynomial.
            pi_gen (List[int]): The coefficients of the quotienting polynomial.

        Raises:
            ValueError: If the quotienting polynomial is empty or not monic.
        """
        if not pi_gen:
            raise ValueError("Quotienting polynomial must be non-empty.")
        if pi_gen[-1] != 1:
            raise ValueError("Quotienting polynomial must be monic.")
        self.pi_generator = pi_gen
        if len(poly) < len(pi_gen):
            poly.extend([0] * (len(pi_gen) - 1 - len(poly)))
            self.element = poly
        else:
            while len(poly) >= len(pi_gen):
                if poly[-1] == 0:
                    poly.pop()
                    continue
                coeff = poly[-1] // pi_gen[-1]
                for i in range(1, len(pi_gen) + 1):
                    poly[-i] -= coeff * pi_gen[-i]
                poly.pop()
            if len(poly) < len(pi_gen) - 1:
                poly.extend([0] * (len(pi_gen) - 1 - len(poly)))
            self.element = poly

    def __repr__(self) -> str:
        """
        Returns a string representation of the QuotientPolynomialRing object.

        Returns:
            str: A string representation of the object.
        """
        return f"QuotientPolynomialRing({self.element}, {self.pi_generator})"

    @staticmethod
    def _check_pi_generators(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> None:
        """
        Checks if two polynomials have the same quotienting polynomial.

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Raises:
            ValueError: If the polynomials do not have the same quotienting polynomial.
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial.")

    @staticmethod
    def Add(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Adds two polynomials in the quotient ring.

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The sum of the two polynomials.
        """
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        op = [x + y for x, y in zip(poly1.element, poly2.element)]
        return QuotientPolynomialRing(op, poly1.pi_generator)

    @staticmethod
    def Sub(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Subtracts one polynomial from another in the quotient ring.

        Args:
            poly1 (QuotientPolynomialRing): The polynomial to subtract from.
            poly2 (QuotientPolynomialRing): The polynomial to subtract.

        Returns:
            QuotientPolynomialRing: The difference of the two polynomials.
        """
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        op = [x - y for x, y in zip(poly1.element, poly2.element)]
        return QuotientPolynomialRing(op, poly1.pi_generator)

    @staticmethod
    def Mul(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Multiplies two polynomials in the quotient ring.

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The product of the two polynomials.
        """
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        op = [0] * (len(poly1.element) * 2 - 1)
        for i in range(len(poly1.element)):
            for j in range(len(poly2.element)):
                op[i + j] += poly1.element[i] * poly2.element[j]
        return QuotientPolynomialRing(op, poly1.pi_generator)

    @staticmethod
    def GCD(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Computes the greatest common divisor (GCD) of two polynomials in the quotient ring.

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The GCD of the two polynomials.
        """
        epsilon = 1e-4
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        if all(abs(coef) < epsilon for coef in poly2.element):
            while poly1.element and poly1.element[-1] == 0:
                poly1.element.pop()
            for i in range(len(poly1.element)):
                poly1.element[i] = int(poly1.element[i] / poly1.element[-1])
            return QuotientPolynomialRing(poly1.element, poly1.pi_generator)
        while poly1.element and poly1.element[-1] == 0:
            poly1.element.pop()
        while poly2.element and poly2.element[-1] == 0:
            poly2.element.pop()
        if len(poly1.element) >= len(poly2.element):
            while len(poly1.element) >= len(poly2.element):
                if poly1.element[-1] == 0:
                    poly1.element.pop()
                    continue
                coeff = poly1.element[-1] / poly2.element[-1]
                for i in range(1, len(poly2.element) + 1):
                    poly1.element[-i] -= coeff * poly2.element[-i]
                poly1.element.pop()
            poly3 = QuotientPolynomialRing(poly1.element, poly1.pi_generator)
            poly4 = QuotientPolynomialRing(poly2.element, poly1.pi_generator)
            return QuotientPolynomialRing.GCD(poly4, poly3)
        else:
            poly3 = QuotientPolynomialRing(poly1.element, poly1.pi_generator)
            poly4 = QuotientPolynomialRing(poly2.element, poly1.pi_generator)
            return QuotientPolynomialRing.GCD(poly4, poly1)

    @staticmethod
    def EGCD(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> Tuple['QuotientPolynomialRing', 'QuotientPolynomialRing', 'QuotientPolynomialRing']:
        """
        Computes the Extended Euclidean Algorithm for two polynomials.

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            Tuple[QuotientPolynomialRing, QuotientPolynomialRing, QuotientPolynomialRing]: 
                - The coefficients (x, y) such that x * poly1 + y * poly2 = gcd(poly1, poly2)
                - The GCD of the two polynomials.
        """
        epsilon = 1e-4
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        if all(abs(coef) < epsilon for coef in poly2.element):
            while poly1.element and poly1.element[-1] == 0:
                poly1.element.pop()
            for i in range(len(poly1.element)):
                poly1.element[i] = int(poly1.element[i] / poly1.element[-1])
            return (QuotientPolynomialRing([1], poly1.pi_generator), 
                    QuotientPolynomialRing([0], poly1.pi_generator),
                    QuotientPolynomialRing(poly1.element, poly1.pi_generator))
        while poly1.element and poly1.element[-1] == 0:
            poly1.element.pop()
        while poly2.element and poly2.element[-1] == 0:
            poly2.element.pop()

        quotient = [0] * (len(poly1.element) - len(poly2.element) + 1)
        remainder = poly1.element[:]
    
        # Get the degree of the divisor
        divisor_degree = len(poly2.element) - 1
        divisor_lead_coeff = poly2.element[-1]
    
        # Perform the division algorithm
        for i in range(len(poly1.element) - len(poly2.element), -1, -1):
            quotient[i] = remainder[i + divisor_degree] // divisor_lead_coeff
            for j in range(divisor_degree + 1):
                remainder[i + j] -= quotient[i] * poly2.element[j]
        
        poly3 = QuotientPolynomialRing(quotient, poly1.pi_generator)
        poly4 = QuotientPolynomialRing(remainder, poly1.pi_generator)
        x, y, gcd = QuotientPolynomialRing.EGCD(poly2, poly4)
        return (y, QuotientPolynomialRing.Sub(x, QuotientPolynomialRing.Mul(y, poly3)), gcd)

    @staticmethod
    def Inv(poly: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Computes the multiplicative inverse of a polynomial in the quotient ring.

        Args:
            poly (QuotientPolynomialRing): The polynomial to invert.

        Returns:
            QuotientPolynomialRing: The inverse of the polynomial.

        Raises:
            Exception: If the polynomial is not invertible.
        """
        new_pigen = poly.pi_generator.copy()
        new_pigen.append(1)
        poly4 = QuotientPolynomialRing(poly.pi_generator, new_pigen)
        poly5 = QuotientPolynomialRing(poly.element, new_pigen)
        poly6 = QuotientPolynomialRing.GCD(poly4, poly5)
        while poly6.element and poly6.element[-1] == 0:
            poly6.element.pop()
        if poly6.element != [1]:
            raise Exception("Not invertible")
        else:
            x, y, gcd = QuotientPolynomialRing.EGCD(poly4, poly5)
            new_pigen.pop()
            return QuotientPolynomialRing(y.element, new_pigen)

def factor(n: int) -> List[Tuple[int, int]]:
    """
    Factors a number into its prime factors and their exponents.

    Args:
        n (int): The number to be factored.

    Returns:
        List[Tuple[int, int]]: A list of tuples where each tuple contains a prime factor and its exponent.
    """
    max_sqrt = int(floor_sqrt(n))
    if n == 1:
        return []
    
    prime = [True] * (max_sqrt + 1)
    prime[0] = False
    prime[1] = False
    factors = []

    if is_prime(n):
        factors.append((n, 1))
        return factors
    
    for i in range(2, max_sqrt + 1):
        if prime[i]:
            if n % i == 0:
                if is_prime(n // i):
                    count_2 = 0
                    k = n // i
                    while n % k == 0:
                        n //= k
                        count_2 += 1
                    factors.append((k, count_2))
                
                count = 0
                while n % i == 0:
                    n //= i
                    count += 1
                factors.append((i, count))
            
            for j in range(i * i, max_sqrt + 1, i):
                prime[j] = False
    
    if is_prime(n):
        factors.append((n, 1))
    
    factors.sort()
    if len(factors) >= 2 and factors[-2][1] == 0:
        factors.pop(-2)
    
    return factors

def euler_phi(n: int) -> int:
    """
    Computes Euler's totient function for a given number.

    Args:
        n (int): The number to compute the totient function for.

    Returns:
        int: The value of Euler's totient function for the given number.
    """
    if n == 1:
        return 1
    
    factors = factor(n)
    result = n
    for prime, _ in factors:
        result *= (prime - 1)
        result //= prime
    
    return result

def is_prime(n: int) -> bool:
    """
    Checks if a number is a prime number using the Miller-Rabin primality test.

    Args:
        n (int): The number to check for primality.

    Returns:
        bool: True if the number is prime, False otherwise.
    """
    if n == 2 or n == 3:
        return True
    if n <= 1 or n % 2 == 0:
        return False
    
    k = 0
    m = n - 1
    while m % 2 == 0:
        k += 1
        m //= 2
    
    for _ in range(10):
        a = random.randint(2, n - 1)
        x = pow(a, m, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(k - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    
    return True

def gen_prime(m: int) -> int:
    """
    Generates a random prime number less than or equal to m.

    Args:
        m (int): The upper bound for the prime number.

    Returns:
        int: A random prime number less than or equal to m.
    """
    while True:
        p = random.randint(2, m)
        if is_prime(p):
            return p

def gen_k_bit_prime(k: int) -> int:
    """
    Generates a random prime number with exactly k bits.

    Args:
        k (int): The number of bits for the prime number.

    Returns:
        int: A random prime number with exactly k bits.
    """
    while True:
        p = random.randint(2**(k - 1), 2**k - 1)
        if is_prime(p):
            return p


from typing import List, Tuple

class QPR_Modulo_P:
    def __init__(self, poly: List[int], pi_gen: List[int], p: int) -> None:
        """
        Initializes a polynomial in the quotient ring modulo a polynomial and a prime.

        Args:
            poly (List[int]): The polynomial coefficients.
            pi_gen (List[int]): The quotienting polynomial (must be monic).
            p (int): The prime modulus.
        
        Raises:
            ValueError: If pi_gen is empty, not monic, or has coefficients >= p.
        """
        if not pi_gen:
            raise ValueError("Quotienting polynomial must be non-empty.")
        if pi_gen[-1] != 1:
            raise ValueError("Quotienting polynomial must be monic.")
        if any(i >= p for i in pi_gen):
            raise ValueError("All coefficients of pi-gen must be in Z_p")
        
        self.pi_generator = pi_gen
        self.p = p
        
        if len(poly) < len(pi_gen):
            self.element = poly
        else:
            while len(poly) >= len(pi_gen):
                if poly[-1] == 0:
                    poly.pop()
                    continue
                coeff = poly[-1]
                for i in range(1, len(pi_gen) + 1):
                    poly[-i] -= coeff * pi_gen[-i]
                poly.pop()
            poly = [(i % p + p) % p for i in poly]
            self.element = poly

    def __repr__(self) -> str:
        return f"QPR_Modulo_P({self.element}, {self.pi_generator}, {self.p})"

    @staticmethod
    def _check_pi_generators(poly1: 'QPR_Modulo_P', poly2: 'QPR_Modulo_P') -> None:
        """
        Checks if two polynomials have the same quotienting polynomial and modulus.

        Args:
            poly1 (QPR_Modulo_P): The first polynomial.
            poly2 (QPR_Modulo_P): The second polynomial.
        
        Raises:
            ValueError: If the polynomials have different quotienting polynomials or moduli.
        """
        if poly1.pi_generator != poly2.pi_generator or poly1.p != poly2.p:
            raise ValueError("Polynomials must have the same quotienting polynomial and modulus.")

    @staticmethod
    def Mul(poly1: 'QPR_Modulo_P', poly2: 'QPR_Modulo_P') -> 'QPR_Modulo_P':
        """
        Multiplies two polynomials in the quotient ring.

        Args:
            poly1 (QPR_Modulo_P): The first polynomial.
            poly2 (QPR_Modulo_P): The second polynomial.

        Returns:
            QPR_Modulo_P: The product of the two polynomials.
        """
        QPR_Modulo_P._check_pi_generators(poly1, poly2)
        op = [0] * (len(poly1.element) + len(poly2.element) - 1)
        for i in range(len(poly1.element)):
            if poly1.element[i] == 0:
                continue
            for j in range(len(poly2.element)):
                if poly2.element[j] == 0:
                    continue
                op[i + j] += poly1.element[i] * poly2.element[j]
                op[i + j] = (op[i + j] + poly1.p) % poly1.p
        return QPR_Modulo_P(op, poly1.pi_generator, poly1.p)

    @staticmethod
    def Mod_Exp(poly: 'QPR_Modulo_P', m: int) -> 'QPR_Modulo_P':
        """
        Computes the exponentiation of a polynomial in the quotient ring.

        Args:
            poly (QPR_Modulo_P): The polynomial to exponentiate.
            m (int): The exponent.

        Returns:
            QPR_Modulo_P: The result of raising the polynomial to the power m.
        """
        if m == 0:
            return QPR_Modulo_P([1], poly.pi_generator, poly.p)
        if m == 1:
            return poly
        
        temp = QPR_Modulo_P.Mod_Exp(poly, m // 2)
        while temp.element[-1] == 0:
            temp.element.pop()
        
        if m % 2 == 0:
            op = QPR_Modulo_P.Mul(temp, temp)
            while op.element[-1] == 0:
                op.element.pop()
            return op
        else:
            op = QPR_Modulo_P.Mul(QPR_Modulo_P.Mul(temp, temp), poly)
            while op.element[-1] == 0:
                op.element.pop()
            return op

def aks_test(n: int) -> bool:
    """
    Performs the AKS primality test to determine if n is a prime number.

    Args:
        n (int): The number to test for primality.

    Returns:
        bool: True if n is prime, False otherwise.
    """
    def find_smallest_r(n: int) -> int:
        """
        Finds the smallest r such that n is not a r-th power modulo any integer.

        Args:
            n (int): The number to find the smallest r for.

        Returns:
            int: The smallest r.
        """

        r = 2
        len_n = n.bit_length()
        bound = 4 * (len_n ** 2)
        while r < n:
            if pair_gcd(n, r) > 1:
                return r
            order = 1
            k = pow(n, order, r)
            while k != 1:
                order += 1
                k *= n
                k %= r
                if order > bound:
                    return r
            r += 1
        return n

    if is_perfect_power(n):
        return False
    
    r = find_smallest_r(n)
    if r == n:
        return True
    
    if pair_gcd(n, r) > 1:
        return False
    
    len_n = n.bit_length()
    pi_gen = [0] * (r + 1)
    pi_gen[0] = -1
    pi_gen[r] = 1
    
    for j in range(1, 2 * len_n * int(floor_sqrt(r)) + 2):
        rhs_poly = [0] * (n % r + 1)
        rhs_poly[0] = j
        rhs_poly[n % r] = 1
        rhs = QPR_Modulo_P(rhs_poly, pi_gen, n)
        
        lhs_poly = [0] * 2
        lhs_poly[0] = j
        lhs_poly[1] = 1
        lhs = QPR_Modulo_P(lhs_poly, pi_gen, n)
        
        l = QPR_Modulo_P.Mod_Exp(lhs, n)
        if l.element != rhs.element:
            return False
    
    return True

def legendre_symbol(a: int, p: int) -> int:
    """
    Compute the Legendre symbol (a/p).

    Args:
        a (int): The numerator of the Legendre symbol.
        p (int): The denominator of the Legendre symbol, a prime number.

    Returns:
        int: The Legendre symbol (a/p), which is either 1, -1, or 0.
    """
    if p == 2:
        return 1
    if a == -1:
        return 1 if p % 4 == 1 else -1
    if a < 0:
        return legendre_symbol(-a, p) * legendre_symbol(-1, p)
    a = a % p
    if a == 2:
        return 1 if (((p ** 2 - 1) // 8) % 2 == 0) else -1
    elif a == 0:
        return 0
    else:
        return 1 if pow(a, (p - 1) // 2, p) == 1 else -1

def jacobi_symbol(a: int, n: int) -> int:
    """
    Compute the Jacobi symbol (a/n).

    Args:
        a (int): The numerator of the Jacobi symbol.
        n (int): The denominator of the Jacobi symbol, which should be an odd positive integer.

    Returns:
        int: The Jacobi symbol (a/n), which is either 1, -1, or 0.
    """
    sigma = 1
    while True:
        a = a % n
        if a == 0:
            if n == 1:
                return sigma
            else:
                return 0
        a_bar = a
        h = 0
        while a_bar % 2 == 0:
            a_bar //= 2
            h += 1
        if h % 2 == 1 and (n % 8 != 1 and n % 8 != 7):
            sigma *= -1
        if a_bar % 4 != 1 and n % 4 != 1:
            sigma *= -1
        a, n = n, a_bar

def get_generator(p: int) -> int:
    """
    Find a generator of the multiplicative group of integers modulo p.

    Args:
        p (int): A prime number.

    Returns:
        int: A generator of the multiplicative group modulo p.
    """
    gamma = 1
    factors = factor(p - 1)
    for factor, exp in factors:
        beta = 1
        while beta == 1:
            alpha = random.randint(2, p - 1)
            beta = pow(alpha, (p - 1) // factor, p)
        gamma = (gamma * pow(alpha, (p - 1) // (factor ** exp), p)) % p
    return gamma

def order(a: int, n: int) -> int:
    """
    Find the order of an element a in the multiplicative group modulo n.

    Args:
        a (int): The element whose order is to be found.
        n (int): The modulus, which should be a prime number.

    Returns:
        int: The order of the element a modulo n.
    """
    l = []
    for i in range(1, floor_sqrt(n - 1) + 1):
        if (n - 1) % i == 0:
            if pow(a, i, n) == 1:
                return i
            l.append((n - 1) // i)
    for i in l:
        if pow(a, i, n) == 1:
            return i
    raise ValueError("Order not found")

def is_smooth(m: int, y: int) -> bool:
    """
    Check if a number m is y-smooth.

    Args:
        m (int): The number to be checked.
        y (int): The smoothness bound.

    Returns:
        bool: True if m is y-smooth, otherwise False.
    """
    if m == 1:
        return True
    for i in range(2, m + 1):
        if m % i == 0:
            if i > y:
                return False
            return is_smooth(m // i, y)
    return True

def dlog_brute_force(x: int, g: int, p: int) -> int:
    """
    Compute discrete logarithm using brute-force method.

    Args:
        x (int): The target value.
        g (int): The base.
        p (int): The modulus, which should be a prime number.

    Returns:
        int: The discrete logarithm of x with base g modulo p.
    """
    o = order(g, p)
    if x == 1:
        return o
    a = g
    for i in range(1, o + 1):
        if g == x:
            return i
        g = (g * a) % p
    raise ValueError("Discrete Logarithm DNE")

def discrete_log(x: int, g: int, p: int) -> int:
    """
    Compute discrete logarithm using baby-step giant-step algorithm.

    Args:
        x (int): The target value.
        g (int): The base.
        p (int): The modulus, which should be a prime number.

    Returns:
        int: The discrete logarithm of x with base g modulo p.
    """
    q = order(g, p)
    m = int(floor_sqrt(q)) + 1
    
    # Precompute baby steps
    baby_steps: Dict[int, int] = {}
    g_powers = 1
    for i in range(m):
        baby_steps[g_powers] = i
        g_powers = (g_powers * g) % p
    
    # Compute inverse of g^m
    g_inv_m = pow(g, -m, p)
    
    current = x
    for j in range(m):
        if current in baby_steps:
            return j * m + baby_steps[current]
        current = (current * g_inv_m) % p
    
    raise ValueError("Discrete Logarithm DNE")

def modular_sqrt_prime(x: int, p: int) -> int:
    """
    Compute modular square root of x modulo a prime p.

    Args:
        x (int): The number whose square root is to be computed.
        p (int): The prime modulus.

    Returns:
        int: A modular square root of x modulo p.
    """
    x = x % p
    if x == 0:
        return 0
    if p == 2:
        return 1
    if legendre_symbol(x, p) == -1:
        raise ValueError("Modular square root DNE")
    if p % 4 == 3:
        op = pow(x, (p + 1) // 4, p)
        if op < p - op:
            return op
        else:
            return p - op
    gamma = random.randint(1, p - 1)
    while legendre_symbol(gamma, p) == 1:
        gamma = random.randint(1, p - 1)
    h = 0
    m = p - 1
    while m % 2 == 0:
        h += 1
        m //= 2
    gamma_2 = pow(gamma, m, p)
    alpha_2 = pow(x, m, p)
    k = discrete_log(alpha_2, gamma_2, p)
    beta = pow(gamma_2, k // 2, p)
    beta *= pow(x, -(m // 2), p)
    beta %= p
    if beta < p - beta:
        return beta
    else:
        return p - beta

def modular_sqrt_prime_power(a: int, p: int, k: int) -> int:
    """
    Compute modular square root of a modulo p^k.

    Args:
        a (int): The number whose square root is to be computed.
        p (int): The prime modulus.
        k (int): The power of the prime.

    Returns:
        int: A modular square root of a modulo p^k.
    """
    if p == 2 and a % 2 == 0:
        return 0
    elif p == 2 and a % 2 == 1:
        return 1
    
    def hensel_lift(a: int, p: int, k: int, b: int) -> int:
        x = b
        for i in range(1, k):
            p_i = p ** i
            x = (x + (a - x * x) * mod_inv(2 * x, p_i)) % (p_i * p)
        return x

    b = modular_sqrt_prime(a, p)
    if b is None:
        raise ValueError("Modular square root DNE")
    op = hensel_lift(a, p, k, b)
    if op < p ** k - op:
        return op
    else:
        return p ** k - op

def modular_sqrt(x: int, z: int) -> int:
    """
    Compute the modular square root of x modulo z using the Chinese Remainder Theorem.

    Args:
        x (int): The number whose square root is to be computed.
        z (int): The modulus.

    Returns:
        int: A modular square root of x modulo z, or the smallest non-negative root.
    """
    def generate_combinations(lst: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Generate all combinations of the given list of pairs.

        Args:
            lst (List[Tuple[int, int]]): List of pairs (a, b) for which to generate combinations.

        Returns:
            List[List[int]]: A list of lists where each inner list is a combination of a's and b's.
        """
        result = [[]]
        for a, b in lst:
            new_result = []
            for combination in result:
                new_result.append(combination + [a])
                new_result.append(combination + [b])
            result = new_result
        return result

    factors = factor(z)
    pre_crt = []
    crt_list = []
    
    for f, exp in factors:
        if exp == 1:
            k = modular_sqrt_prime(x, f)
            pre_crt.append((k, f - k))
            crt_list.append(f)
        else:
            k = modular_sqrt_prime_power(x, f, exp)
            pre_crt.append((k, f ** exp - k))
            crt_list.append(f ** exp)

    combinations = generate_combinations(pre_crt)
    min_root = z
    
    for comb in combinations:
        root = crt(comb, crt_list)
        if root < min_root:
            min_root = root

    return min_root 

def probabilistic_dlog(x: int, g: int, p: int) -> Optional[int]:
    """
    Compute the discrete logarithm of x with base g modulo p using a probabilistic approach.

    Args:
        x (int): The target value for which the discrete logarithm is to be computed.
        g (int): The base of the discrete logarithm.
        p (int): The modulus, which should be a prime number.

    Returns:
        Optional[int]: The discrete logarithm of x with base g modulo p if found, otherwise None.
    """
    Q = (p - 1) // 2  

    def process(y: int, a: int, b: int) -> Tuple[int, int, int]:
        """
        Update y, a, and b based on modular operations.

        Args:
            y (int): Current value of y.
            a (int): Current value of a.
            b (int): Current value of b.

        Returns:
            Tuple[int, int, int]: Updated values of y, a, and b.
        """
        remainder = y % 3

        if remainder == 0:
            y = y * g % p
            a = (a + 1) % Q

        elif remainder == 1:
            y = y * x % p
            b = (b + 1) % Q

        elif remainder == 2:
            y = y * y % p
            a = a * 2 % Q
            b = b * 2 % Q

        return y, a, b

    def check(g: int, h: int, p: int, x: int) -> bool:
        """
        Verify if g^x ≡ h (mod p).

        Args:
            g (int): The base of the logarithm.
            h (int): The target value.
            p (int): The modulus.
            x (int): The candidate for the discrete logarithm.

        Returns:
            bool: True if g^x ≡ h (mod p), otherwise False.
        """
        return pow(g, x, p) == h

    try:
        initial_y = g * x % p
        initial_a = 1
        initial_b = 1

        current_y = initial_y
        current_a = initial_a
        current_b = initial_b

        for _ in range(1, p):
            current_y, current_a, current_b = process(current_y, current_a, current_b)
            initial_y, initial_a, initial_b = process(initial_y, initial_a, initial_b)
            initial_y, initial_a, initial_b = process(initial_y, initial_a, initial_b)

            if current_y == initial_y:
                break

        numerator = (current_a - initial_a) % Q
        denominator = (initial_b - current_b) % Q

        result = (mod_inv(denominator, Q) * numerator) % Q

        if check(g, x, p, result):
            return result
        else:
            return discrete_log(x, g, p)

    except Exception:
        return discrete_log(x, g, p)

def probabilistic_factor(N: int) -> List[Tuple[int, int]]:
    """
    Factorizes a large integer N using the Quadratic Sieve algorithm.

    Args:
        N (int): The integer to factorize.

    Returns:
        List[Tuple[int, int]]: A list of tuples where each tuple contains a prime factor and its exponent.

    Raises:
        ValueError: If no factors are found and the factorization fails.
    """
    
    if is_prime(N):
        return [(N, 1)]
    
    factor_list = []

    def factor_small_numbers(N: int) -> List[Tuple[int, int]]:
        """
        Factorizes N for small numbers using trial division up to 1000.

        Args:
            N (int): The integer to factorize.

        Returns:
            List[Tuple[int, int]]: A list of tuples where each tuple contains a prime factor and its exponent.
        """
        factor_list = []
        for i in range(2, 1000):
            if is_prime(i):
                count = 0
                while N % i == 0:
                    count += 1
                    N //= i
                if count > 0:
                    factor_list.append((i, count))
        return factor_list

    def gaussian_elimination(matrix: List[List[int]]) -> Tuple[List[bool], List[List[int]]]:
        """
        Performs Gaussian elimination on a matrix over GF(2).

        Args:
            matrix (List[List[int]]): The matrix to perform elimination on.

        Returns:
            Tuple[List[bool], List[List[int]]]: A tuple where the first element is a list indicating whether each row is marked,
                                                and the second element is the transformed matrix.
        """
        marks = [False] * len(matrix)
        for col in range(len(matrix[0])):
            for row in range(len(matrix)):
                if matrix[row][col] == 1:
                    marks[row] = True
                    for k in range(col):
                        if matrix[row][k] == 1:
                            for r in range(len(matrix)):
                                matrix[r][k] = (matrix[r][k] + matrix[row][k]) % 2
                    for k in range(col + 1, len(matrix[0])):
                        if matrix[row][k] == 1:
                            for r in range(len(matrix)):
                                matrix[r][k] = (matrix[r][k] + matrix[row][k]) % 2
                    break
        return marks, matrix

    def dependent_columns(row: List[int]) -> List[int]:
        """
        Finds the indices of columns in a row where the value is 1.

        Args:
            row (List[int]): The row to check.

        Returns:
            List[int]: A list of column indices where the value is 1.
        """
        return [i for i in range(len(row)) if row[i] == 1]

    def add_rows(new_row: int, current_row: List[int]) -> List[int]:
        """
        Adds two rows in the matrix using XOR operation.

        Args:
            new_row (int): The index of the new row.
            current_row (List[int]): The current row to add to.

        Returns:
            List[int]: The resulting row after addition.
        """
        return [current_row[i] ^ matrix[new_row][i] for i in range(len(matrix[new_row]))]

    def is_row_dependent(cols: List[int], row: List[int]) -> bool:
        """
        Checks if a row is dependent on a given set of columns.

        Args:
            cols (List[int]): The list of column indices to check.
            row (List[int]): The row to check for dependency.

        Returns:
            bool: True if the row is dependent on the given columns, False otherwise.
        """
        return any(row[i] == 1 for i in cols)

    def find_dependent_rows(row: int) -> List[List[int]]:
        """
        Finds dependent rows for a given row in the matrix.

        Args:
            row (int): The index of the row to find dependencies for.

        Returns:
            List[List[int]]: A list of lists where each inner list represents a set of dependent rows.
        """
        dependencies = []
        dependent_cols = dependent_columns(matrix[row])
        current_rows = [row]
        current_sum = matrix[row][:]
        for i in range(len(matrix)):
            if i == row:
                continue
            if is_row_dependent(dependent_cols, matrix[i]):
                current_rows.append(i)
                current_sum = add_rows(i, current_sum)
                if sum(current_sum) == 0:
                    dependencies.append(current_rows[:])
        return dependencies

    def test_dependency(dep_rows: List[List[int]]) -> int:
        """
        Tests the dependency of rows to find a non-trivial factor.

        Args:
            dep_rows (List[List[int]]): A list of dependent rows.

        Returns:
            int: A factor of N found from the dependencies.
        """
        x = y = 1
        for row in dep_rows:
            x *= smooth_values[row][0]
            y *= smooth_values[row][1]
        return extended_gcd(x - floor_sqrt(y), N)[0]

    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """
        Computes the extended greatest common divisor of a and b.

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            Tuple[int, int, int]: A tuple where the first element is the GCD, and the second and third elements are the
                                  coefficients of the extended GCD.
        """
        prev_x, x = 1, 0
        prev_y, y = 0, 1
        while b:
            q, r = divmod(a, b)
            x, prev_x = prev_x - q * x, x
            y, prev_y = prev_y - q * y, y
            a, b = b, r
        return a, prev_x, prev_y

    def create_prime_base(n: int, B: int) -> List[int]:
        """
        Creates a base of B primes that are valid for the factorization.

        Args:
            n (int): The number to create the base for.
            B (int): The number of primes to include in the base.

        Returns:
            List[int]: A list of prime numbers.
        """
        primes = []
        i = 2
        while len(primes) < B:
            if legendre_symbol(n, i) == 1:
                primes.append(i)
            i += 1
            while not is_prime(i):
                i += 1
        return primes

    def polynomial(x: int, a: int, b: int, n: int) -> int:
        """
        Computes the value of the quadratic polynomial.

        Args:
            x (int): The value of x in the polynomial.
            a (int): The coefficient a in the polynomial.
            b (int): The coefficient b in the polynomial.
            n (int): The constant n in the polynomial.

        Returns:
            int: The result of the polynomial.
        """
        return ((a * x + b) ** 2) - n

    def solve_polynomial(a: int, b: int, n: int) -> List[List[int]]:
        """
        Solves the polynomial for a given base of primes.

        Args:
            a (int): The coefficient a in the polynomial.
            b (int): The coefficient b in the polynomial.
            n (int): The constant n in the polynomial.

        Returns:
            List[List[int]]: A list of start values for each base prime.
        """
        start_values = []
        for prime in base_primes:
            inv_a = mod_inv(a, prime)
            root1 = modular_sqrt_prime(n, prime)
            root2 = (-1 * root1) % prime
            start1 = (inv_a * (root1 - b)) % prime
            start2 = (inv_a * (root2 - b)) % prime
            start_values.append([start1, start2])
        return start_values

    def trial_division(n: int, base: List[int]) -> List[int]:
        """
        Performs trial division of n by base primes.

        Args:
            n (int): The number to divide.
            base (List[int]): The list of base primes.

        Returns:
            List[int]: A list of exponents for each base prime.
        """
        result = [0] * len(base)
        if n > 0:
            for i in range(len(base)):
                while n % base[i] == 0:
                    n //= base[i]
                    result[i] = (result[i] + 1) % 2
        return result

    if N < 10**12:
        return factor_small_numbers(N)

    a = 1
    b = floor_sqrt(N) + 1
    bound = 50
    base_primes = create_prime_base(N, bound)
    required = euler_phi(base_primes[-1]) + 1

    sieve_start = 0
    sieve_stop = 0
    sieve_interval = 100000

    matrix = []
    smooth_values = []
    start_values = solve_polynomial(a, b, N)
    seen_combinations = set()

    while len(smooth_values) < required:
        sieve_start = sieve_stop
        sieve_stop += sieve_interval
        interval_values = [polynomial(x, a, b, N) for x in range(sieve_start, sieve_stop)]

        for p in range(len(base_primes)):
            t = start_values[p][0]
            while start_values[p][0] < sieve_start + sieve_interval:
                while interval_values[start_values[p][0] - sieve_start] % base_primes[p] == 0:
                    interval_values[start_values[p][0] - sieve_start] //= base_primes[p]
                start_values[p][0] += base_primes[p]
            if start_values[p][1] != t:
                while start_values[p][1] < sieve_start + sieve_interval:
                    while interval_values[start_values[p][1] - sieve_start] % base_primes[p] == 0:
                        interval_values[start_values[p][1] - sieve_start] //= base_primes[p]
                    start_values[p][1] += base_primes[p]
        for i in range(sieve_interval):
            if interval_values[i] == 1:
                x = sieve_start + i
                y = polynomial(x, a, b, N)
                exp = trial_division(y, base_primes)
                if tuple(exp) not in seen_combinations:
                    smooth_values.append(((a * x) + b, y))
                    matrix.append(exp)
                    seen_combinations.add(tuple(exp))

    row_marks, matrix = gaussian_elimination(matrix)
    
    for i in range(len(row_marks)):
        if not row_marks[i]:
            dependencies = find_dependent_rows(i)
            for dep in dependencies:
                gcd_result = test_dependency(dep)
                if gcd_result != 1 and gcd_result != N:
                    print(gcd_result, N // gcd_result)
                    if gcd_result < 10**12:
                        factor_list.extend(factor_small_numbers(gcd_result))
                    else:
                        factor_list.extend(probabilistic_factor(gcd_result))
                    if (N // gcd_result) < 10**12:
                        factor_list.extend(factor_small_numbers(N // gcd_result))
                    else:
                        factor_list.extend(probabilistic_factor(N // gcd_result))
                    return factor_list

    raise ValueError("Factorization failed or no factors found.")


