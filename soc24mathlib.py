import random
def pair_gcd(a, b): #returns gcd(a,b) [op:int, ip:(int, int)]
    if(b==0):
        return a
    else:
        return pair_gcd(b, a%b)

def pair_egcd(a, b): # returns (x, y, gcd(a,b)), where ax + by = gcd(a,b)   [op:(int, int,int), ip:(int, int)]
    if(b==0):
        return 1, 0, a
    else:
        x, y, gcd = pair_egcd(b, a%b)
        return y , x - y*(a//b), gcd 

def are_relatively_prime(a,b): #returns true if a and b are co prime, false other wise [op: bool, ip: (int, int)]
    return pair_gcd(a,b) ==1

def gcd(*input): # returns gcd(a_1, a_2, ..., a_n) [ip:(*int), op:int]
    if(len(input)==2):
        return pair_gcd(input)
    else:
        op =0
        for i in input:
            op = pair_gcd(i, op)
        return op

def pair_lcm(a, b): # returns lcm(a,b) [op:int, ip:(int, int)]
    return (a//pair_gcd(a,b))*b

def lcm(*input): # returns lcm(a_1, a_2,...) [op:int, ip:(int, int,...)]
    if(len(input)==2):
        return pair_lcm(input)
    else:
        op=1
        for i in input:
            op= pair_lcm(op, i)
        return op
    
def mod_inv(a, n): #return modular inverse of a mod n [op:int, ip:(int, int)]
  if (are_relatively_prime(a, n) != 1):
      raise Exception(f"{a} and {n} are not coprime")
  x,y, g = pair_egcd(a,n)
  return x%n

def crt(a, n): #returns an x that satisfies all simultaneous congruence [op:int, ip:(int[], int[])]
    pi=1
    for i in n:
        pi *= i
    op=0
    for i in range (0,len(n)):
        op+= ((pi//n[i])%pi)*(mod_inv(pi//n[i], n[i])%pi)*(a[i]%pi) % pi
    return op%pi

def pow(a,m,n): #returns a to the power m modulo n [op:int, ip:(int, int, int)]
    if(m==1):
        return a%n
    if(m==0):
        return 1
    temp= pow(a, m//2, n)%n
    if(m%2==0):
        return ((temp)**2)%n
    else:
        return ((temp)**2)*(a%n)%n

def is_quadratic_residue_prime(a,p): # returns 0 if a,p aren't coprime, 1 if a is a quadratic residue else -1 [op:int, ip:(int, int)]
    if(p==2):
        return 1
    if(are_relatively_prime(a,p)==0):
        return 0
    else:
        if(pow(a,(p-1)/2,p) ==1):
            return 1
        else:
            return -1

def is_quadratic_residue_prime_power(a,p,e): # returns 0 if a,p^e aren't coprime, 1 if a is a quadratic residue else -1 [op:int, ip:(int, int)]
    if(p==1):
        return 1
    if(are_relatively_prime(a,p)==0):
        return 0
    else:
        if(pow(a,(p-1)/2,p) ==1):
            return 1
        else:
            return -1

def floor_sqrt(n): # returns floor(sqrt(n)) [op:int, ip:int]
    len_n = n.bit_length()
    k= (len_n-1)//2
    m=2**k
    two_raised_to_i = 2**(k-1)
    for i in range(k-1, -1, -1):
        if((m+two_raised_to_i)**2<=n):
            m= m+two_raised_to_i
        two_raised_to_i /= 2
    return m

def floor_k_root(n, k): # returns floor(n^(1/k)) [op:int, ip:(int, int)]
    len_n = n.bit_length()
    m=2**((len_n-1)//k)
    two_raised_to_i = 2**((len_n-1)//(2*k))
    for i in range((len_n-1)//(2*k), -1, -1):
        if((m+two_raised_to_i)**k<=n):
            m= m+two_raised_to_i
        two_raised_to_i /= 2
    return int(m)

def is_perfect_power(n): # returns true if n is a perfect power, false otherwise [op:bool, ip:int]
    len_n = n.bit_length()
    for i in range(2, len_n):
        k = floor_k_root(n, i)
        if(k**i==n):
            return True
    return False

class QuotientPolynomialRing:
    def __init__(self, poly: list[int], pi_gen: list[int]) -> None:
        if not pi_gen:
            raise ValueError("Quotienting polynomial must be non-empty.")
        if pi_gen[-1] != 1:
            raise ValueError("Quotienting polynomial must be monic.")
        self.pi_generator = pi_gen
        if(len(poly)<len(pi_gen)):
            poly.extend([0] * (len(pi_gen) - 1 - len(poly)))
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
            if len(poly) < len(pi_gen) - 1:
                poly.extend([0] * (len(pi_gen) - 1 - len(poly)))
            self.element = poly                       
        
    def __repr__(self):
        return f"QuotientPolynomialRing({self.element}, {self.pi_generator})"

    @staticmethod
    def _check_pi_generators(poly1, poly2):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same quotienting polynomial.")

    @staticmethod
    def Add(poly1, poly2):
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        op=[]
        for i in range (0,len(poly1.element)):
            op.append(poly1.element[i]+poly2.element[i])
        return QuotientPolynomialRing(op, poly1.pi_generator)

    @staticmethod
    def Sub(poly1, poly2):
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        op=[]
        for i in range (0, len(poly1.element)):
            op.append(poly1.element[i]-poly2.element[i])
        return QuotientPolynomialRing(op, poly1.pi_generator)

    @staticmethod
    def Mul(poly1, poly2):
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        op=[0]*(len(poly1.element)*2-1)
        for i in range (0, len(poly1.element)):
            for j in range (0, len(poly2.element)):
                op[i+j] += poly1.element[i]*poly2.element[j]
        return QuotientPolynomialRing(op, poly1.pi_generator)
    
    @staticmethod
    def quotient(poly1, poly2):
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        quotient = []
        while poly1.element[-1]==0:
            poly1.element.pop()
        while poly2.element[-1]==0:
            poly2.element.pop()  
        remainder=poly1.element      
        while len(remainder) >= len(poly2.element):
            if remainder[-1] == 0:
                remainder.pop()
                continue
            
            lead_coef_ratio = remainder[-1] // poly2.element[-1]
            degree_diff = len(remainder) - len(poly2.element)
            
            current_quotient = [0] * degree_diff + [lead_coef_ratio]
            quotient = current_quotient + quotient
            coeff=remainder[-1]
            for i in range(1, len(poly2.element)+1):
                remainder[-i]=coeff*poly2.element[-i]
            remainder.pop()  
        if not quotient:
            quotient = [0]        
        return QuotientPolynomialRing(quotient, poly1.pi_generator)       
    
    @staticmethod
    def polynomial_mod(poly1, poly2):
        while poly1.element[-1]==0:
            poly1.element.pop()
        while poly2.element[-1]==0:
            poly2.element.pop()
        while len(poly1.element) >= len(poly2.element):
                if poly1.element[-1] == 0:
                    poly1.element.pop()
                    continue
                coeff = poly1.element[-1]/poly2.element[-1]
                for i in range(1, len(poly2.element) + 1):
                    poly1.element[-i] -= coeff * poly2.element[-i]
                poly1.element.pop()
        return QuotientPolynomialRing(poly1.element, poly1.pi_generator)

    @staticmethod
    def GCD(poly1, poly2):
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        if poly2 == QuotientPolynomialRing([0], poly1.pi_generator):
            return poly1
        else:
            return QuotientPolynomialRing.GCD(poly2, QuotientPolynomialRing.polynomial_mod(poly1, poly2))

    @staticmethod
    def egcd(poly1, poly2):
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        if poly2.element == [0]*len(poly2.element):
            return QuotientPolynomialRing(QuotientPolynomialRing([1], poly1.pi_generator),QuotientPolynomialRing([0], poly1.pi_generator), poly1)
        else:
            x,y,g= QuotientPolynomialRing.egcd(poly2, QuotientPolynomialRing.polynomial_mod(poly1.element, poly2.element))
            return y, QuotientPolynomialRing.Sub(x, QuotientPolynomialRing.Mul(y,QuotientPolynomialRing.quotient(poly1, poly2))), g

    @staticmethod
    def Inv(poly):
        x, y, g = QuotientPolynomialRing.egcd(poly, poly.pi_generator)
        if g != QuotientPolynomialRing([1], poly.pi_generator):
            raise ValueError("Polynomial is not invertible in the ring.")
        return QuotientPolynomialRing(x, poly.pi_generator)

def factor(n):
    max= int(floor_sqrt(n))
    is_prime=[True]*(max+1)
    is_prime[0]=False
    is_prime[1]=False
    op=[]
    if(n==1):
        return op
    for i in range(2, max+1):
        if(is_prime[i]):
            if(n%i==0):
                count=0
                while(n%i==0):
                    n//=i
                    count+=1
                op.append((i, count))
            for j in range(i*i, max+1, i):
                is_prime[j]=False
    if(len(op)==0):
        op.append((n, 1))
    return op
    
def euler_phi(n):
    if(n==1):
        return 1
    f=factor(n)
    p=n
    for i in range(0,len(f)):
        p *= (f[i][0]-1)
        p //= f[i][0]
    return p

def is_prime(n):
    if(n==2 or n==3):
        return True
    if(n<=1 or n%2==0):
        return False
    k=0
    m=n-1
    while m%2==0:
        k+=1
        m//=2
    
    for i in range(0, 10):
        a=random.randint(2, n-1)
        x=pow(a, m, n)
        if x==1 or x==n-1:
            continue
        for i in range(0,k-1):
            x=(x**2)%n
            if x==n-1:
                break
        else:
            return False
    return True