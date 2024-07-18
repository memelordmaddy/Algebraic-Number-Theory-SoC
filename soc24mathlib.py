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
    if(m<0):
        return mod_inv(pow(a,-m,n),n)
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
    return int(m)

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

def natural_log(n):
    e=2.718281828459045
    e_pow=e
    k=1
    while e_pow*e<=n:
        e_pow*=e
        k+=1
    n/=e_pow
    n-=1
    natural_log=0
    n_cp=n
    for i in range(1,10):
        if i%2==0:
            natural_log-=n/i
        else:
            natural_log+=n/i
        n*=n_cp
    return natural_log+k

def exp(n):
    f=1
    op=1
    n_cp=n
    for i in range(1,100):
        f*=i
        op += n/f
        n*=n_cp
    return op

class QuotientPolynomialRing:
    def __init__(self, poly, pi_gen) -> None:
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
                coeff = poly[-1]// pi_gen[-1]
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
    def GCD(poly1, poly2):
        #print(poly1, poly2)
        epsilon=1e-4
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        flag=True
        for i in range (0, len(poly2.element)):
            if(poly2.element[i]>epsilon or poly2.element[i]<-epsilon):
                flag=False
                break
        if(flag):
            while poly1.element and poly1.element[-1]==0:
                poly1.element.pop()
            for i in range(0,len(poly1.element)):
                poly1.element[i]=int(poly1.element[i]/poly1.element[-1])
            return QuotientPolynomialRing(poly1.element, poly1.pi_generator)          
        while poly1.element and poly1.element[-1]==0:
            poly1.element.pop()
        while poly2.element and poly2.element[-1]==0:
            poly2.element.pop()
        if(len(poly1.element)>=len(poly2.element)):
            while len(poly1.element) >= len(poly2.element):
                if poly1.element[-1] == 0:
                    poly1.element.pop()
                    continue
                coeff = poly1.element[-1]/poly2.element[-1]
                for i in range(1, len(poly2.element) + 1):
                    poly1.element[-i] -= coeff * poly2.element[-i]
                poly1.element.pop()
            poly3= QuotientPolynomialRing(poly1.element, poly1.pi_generator)
            poly4= QuotientPolynomialRing(poly2.element, poly1.pi_generator)
            return QuotientPolynomialRing.GCD(poly4, poly3)
        else:
            poly3= QuotientPolynomialRing(poly1.element, poly1.pi_generator)
            poly4= QuotientPolynomialRing(poly2.element, poly1.pi_generator)
            return  QuotientPolynomialRing.GCD(poly4,poly1)
        
    @staticmethod
    def EGCD(poly1, poly2):
        epsilon=1e-4
        QuotientPolynomialRing._check_pi_generators(poly1, poly2)
        flag=True
        for i in range (0, len(poly2.element)):
            if(poly2.element[i]>epsilon or poly2.element[i]<-epsilon):
                flag=False
                break
        if(flag):
            while poly1.element and poly1.element[-1]==0:
                poly1.element.pop()
            for i in range(0,len(poly1.element)):
                poly1.element[i]=int(poly1.element[i]/poly1.element[-1])
            return QuotientPolynomialRing([1],poly1.pi_generator),QuotientPolynomialRing([0], poly1.pi_generator),QuotientPolynomialRing(poly1.element, poly1.pi_generator)
        while poly1.element and poly1.element[-1]==0:
            poly1.element.pop()
        while poly2.element and poly2.element[-1]==0:
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
        
        poly3= QuotientPolynomialRing(quotient, poly1.pi_generator)
        poly4= QuotientPolynomialRing(remainder, poly1.pi_generator)
        x,y,gcd = QuotientPolynomialRing.EGCD(poly2, poly4)
        return y, QuotientPolynomialRing.Sub(x, QuotientPolynomialRing.Mul(y, poly3)), gcd
    
    @staticmethod
    def Inv(poly):
        new_pigen=poly.pi_generator.copy()
        new_pigen.append(1)
        poly4= QuotientPolynomialRing(poly.pi_generator, new_pigen)
        #print(poly4)
        poly5= QuotientPolynomialRing(poly.element, new_pigen)
        poly6=QuotientPolynomialRing.GCD(poly4, poly5)
        #print(poly6.element)
        while poly6.element and poly6.element[-1]==0:
            poly6.element.pop()
        if(poly6.element!=[1]):
            #print(QuotientPolynomialRing.GCD(poly4, poly5))
            raise Exception("Not invertible")
        else:
            x,y,gcd = QuotientPolynomialRing.EGCD(poly4, poly5)
            #print(poly.pi_generator)
            #print(y.element)
            new_pigen.pop()
            #print(new_pigen)
            return QuotientPolynomialRing(y.element, new_pigen)
            
         
def factor(n):
    n_cp=n
    max= int(floor_sqrt(n))
    if(n==1):
        return []
    prime=[True]*(max+1)
    prime[0]=False
    prime[1]=False
    op=[]
    if(n==1):
        return op
    if(is_prime(n)):
        op.append((n, 1))
        return op
    for i in range(2, max+1):
        if(prime[i]):
            if(n%i==0):
                if(is_prime(n//i) ):
                    count_2=0
                    k=n//i
                    while(n%k==0):
                        n//=k
                        count_2+=1
                    op.append((k, count_2))
                count=0
                while(n%i==0):
                    n//=i
                    count+=1
                op.append((i, count))
            for j in range(i*i, max+1, i):
                prime[j]=False
    op.sort()
    if(len(op)>=2 and op[-2][1]==0):
        op.pop(-2)
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

def gen_prime(m):
    while True:
        p=random.randint(2,m)
        if(is_prime(p)):
            return p
        
def gen_k_bit_prime(k):
    while True:
        p=random.randint(2**(k-1),2**k-1)
        if(is_prime(p)):
            return p
        
class QPR_Modulo_P:
    def __init__(self, poly:list[int], pi_gen: list[int], p:int) -> None:
        if not pi_gen:
            raise ValueError("Quotienting polynomial must be non-empty.")
        if pi_gen[-1] != 1:
            raise ValueError("Quotienting polynomial must be monic.")
        for i in pi_gen:
            if i>=p:
                raise ValueError("All coefficients of pi-gen must be in Z_p")
        self.pi_generator = pi_gen
        self.p = p
        if(len(poly)<len(pi_gen)):
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
            for i in range(0, len(poly)):
                poly[i]%=p
                poly[i]= (poly[i]+p)%p
            self.element = poly

    def __repr__(self):
        return f"QPR_Modulo_P({self.element}, {self.pi_generator}, {self.p})"
    
    @staticmethod
    def _check_pi_generators(poly1, poly2):
        if poly1.pi_generator != poly2.pi_generator or poly1.p != poly2.p:
            raise ValueError("Polynomials must have the same quotienting polynomial and p.")

    @staticmethod
    def Mul(poly1, poly2):
        QPR_Modulo_P._check_pi_generators(poly1, poly2)
        op=[0]*(len(poly1.element) + len(poly2.element)-1)
        for i in range (0, len(poly1.element)):
            if(poly1.element[i]==0):
                continue
            for j in range (0, len(poly2.element)):
                if(poly2.element[j]==0):
                    continue
                op[i+j] += poly1.element[i]*poly2.element[j]
                op[i+j]= (op[i+j]+poly1.p)%poly1.p
        return QPR_Modulo_P(op, poly1.pi_generator, poly1.p)
    
    @staticmethod
    def Mod_Exp(poly, m):
        #print(m)
        if(m==0):
            return QPR_Modulo_P([1], poly.pi_generator, poly.p)
        if(m==1):
            return poly
        temp= QPR_Modulo_P.Mod_Exp(poly, m//2)
        while temp.element[-1]==0:
            temp.element.pop()

        #print(temp)
        if(m%2==0):
            op= QPR_Modulo_P.Mul(temp, temp)
            while op.element[-1]==0:
                op.element.pop()
            return op
        else:
            op= QPR_Modulo_P.Mul(QPR_Modulo_P.Mul(temp, temp), poly)
            while op.element[-1]==0:
                op.element.pop()
            return op
    
def find_smallest_r(n):
    r = 2
    len_n = n.bit_length()
    bound= 4*(len_n**2)
    while r < n:
        if pair_gcd(n, r) > 1:
            return r
        order = 1
        k=pow(n, order, r)
        while k != 1:
            order += 1
            k*=n
            k%=r
            if order > bound:
                return r
        r += 1
    return n

def aks_test(n):
    if(is_perfect_power(n)):
        return False
    r=find_smallest_r(n)
    if(r==n):
        return True
    if pair_gcd(n, r) > 1:
        return False
    len_n = n.bit_length()
    pi_gen=[0]*(r+1)
    pi_gen[0]=-1
    pi_gen[r]=1
    for j in range(1, 2*len_n*int(floor_sqrt(r))+2):
        print(j)
        rhs_poly=[0]*(n%r+1)
        rhs_poly[0]=j
        rhs_poly[n%r]=1
        rhs=QPR_Modulo_P(rhs_poly, pi_gen, n)
        #print(rhs)
        lhs_poly=[0]*(2)
        lhs_poly[0]=j
        lhs_poly[1]=1
        lhs=QPR_Modulo_P(lhs_poly, pi_gen, n)
        #print(lhs)
        l=QPR_Modulo_P.Mod_Exp(lhs, n)
        if l.element!=rhs.element:
            #print(l.element, rhs.element)

            return False
    return True

def legendre_symbol(a, p):
    if(p==2):
        return 1
    if(a==-1):
        if(p%4==1):
            return 1
        else:
            return -1
    if(a<0):
        return legendre_symbol(-a, p)*legendre_symbol(-1, p)
    a= a%p
    if(a==2):
        if(((p**2-1)//8) %2==0):
            return 1
        else:
            return -1
    elif(a==0):
        return 0
    else:
        if( pow(a, (p-1)//2, p)==1):
            return 1
        else:
            return -1
    
def jacobi_symbol(a,n):
    sigma=1
    while True:
        a=a%n
        if(a==0):
            if(n==1):
                return sigma
            else:
                return 0
        a_bar=a
        h=0
        while a_bar%2==0:
            a_bar= a_bar//2
            h+=1
        if(h%2==1 and (n%8!=1 and n%8!=7)):
            sigma*=-1
        if(a_bar%4 !=1 and n%4!=1):
            sigma*=-1
        a=n
        n=a_bar



def get_generator(p):
    gamma=1
    factors = factor(p-1)
    for i in range (0, len(factors)):
        beta=1
        while beta==1:
            alpha= random.randint(2, p-1)
            beta= pow(alpha, (p-1)//factors[i][0], p)
        gamma= (gamma*pow(alpha,(p-1)/(factors[i][0]**factors[i][1]),p))%p
    return gamma

def order(a,n):
    l=[]
    for i in range(1,int(floor_sqrt(n-1))):
        if((n-1)%i==0):
            if(pow(a,i,n)==1):
                return i
            l.append((n-1)//i)
    for i in l:
        if(pow(a,i,n)==1):
            return i
        
def is_smooth(m,y):
    if(m==1):
        return True
    for i in range(2, m+1):
        if(m%i==0):
            if(i>y):
                return False
            return is_smooth(m//i, y)
    return True

def dlog_brute_force(x,g,p):
    o= order(g,p)
    if(x==1):
        return o
    a=g
    for i in range (1,o+1):
        if(g==x):
            return i
        g= (g*a)%p
    raise ValueError("Discrete Logarithm DNE")

def dlog_baby_giant(x, g, p):
    q = order(g, p)
    m = int(floor_sqrt(q)) + 1
    
    baby_steps = {}
    for i in range(m):
        baby_steps[pow(g, i, p)] = i
    
    g_inv_m = pow(g, m, p)
    g_inv_m= mod_inv(g_inv_m,p)
    
    current = x
    for j in range(m):
        if current in baby_steps:
            return j * m + baby_steps[current]
        current = (current * g_inv_m) % p
    
    raise ValueError("Discrete Logarithm DNE")

def discrete_log(x,g,p): # OPTIMIZE USING RDL
    return dlog_baby_giant(x,g,p)

def modular_sqrt_prime(x,p):
    x=x%p
    if x==0:
        return 0
    if(p==2):
        return 1
    if(legendre_symbol(x,p)==-1):
        raise ValueError("modular square root DNE")
    if(p%4==3):
        op= pow(x,(p+1)/4,p)
        if op<p-op:
            return op
        else:
            return p-op
    gamma= random.randint(1,p-1)
    while legendre_symbol(gamma,p)==1:
        gamma=random.randint(1,p-1)
    h=0
    m=p-1
    while m%2==0:
        h+=1
        m/=2
    gamma_2= pow(gamma,m,p)
    alpha_2= pow(x,m,p)
    k=discrete_log(alpha_2, gamma_2,p)
    beta=pow(gamma_2,k/2,p)
    beta *= pow(x,-(m//2),p)
    beta%=p
    if beta<p-beta:
        return beta
    else:
        return p-beta


def modular_sqrt_prime_power(a, p, k):    
    def hensel_lift(a, p, k, b):
        x = b
        for i in range(1, k):
            p_i = p**i
            x = (x + (a - x*x) * pow(2*x, -1, p_i)) % (p_i * p)
        return x
    b = modular_sqrt_prime(a, p)
    if b is None:
        raise ValueError("modular square root DNE")
    op= hensel_lift(a, p, k, b)
    if op<p**k-op:
        return op
    else:
        return p**k-op    

def modular_sqrt(x,z):
    def generate_combinations(lst):
        result = [[]]
        for a, b in lst:
            new_result = []
            for combination in result:
                new_result.append(combination + [a])
                new_result.append(combination + [b])
            result = new_result
        return result
    
    factors= factor(z)
    pre_crt=[]
    crt_list=[]
    for i in range (0, len(factors)):
        if(factors[i][1]==1):
            k= modular_sqrt_prime(x, factors[i][0])
            pre_crt.append((k, factors[i][0]-k))
            crt_list.append(factors[i][0])                           
        else:
            k= modular_sqrt_prime_power(x, factors[i][0], factors[i][1])
            pre_crt.append((k, factors[i][0]**factors[i][1]-k))
            crt_list.append(factors[i][0]**factors[i][1])
    
    combinations= generate_combinations(pre_crt)
    possible_roots=[]
    min_root= z
    for i in range (0, len(combinations)):
        possible_roots.append(crt(combinations[i], crt_list))
        if(possible_roots[i]<min_root):
            min_root= possible_roots[i]

    return min_root     

def gaussian_elimination(A, b,p):
    n = len(A[0])

    # Forward elimination
    for i in range(n):
        # Pivot for maximum element in column i to avoid division by zero
        max_row = max(range(i, n), key=lambda r: abs(A[i][r]))
        for k in range(n):
            A[k][i], A[k][max_row] = A[k][max_row], A[k][i]
        b[i], b[max_row] = b[max_row], b[i]

        # Eliminate entries below i
        for j in range(i + 1, n):
            if A[i][j] == 0:
                continue
            factor = (A[i][j] *  mod_inv(A[i][i],p))%p
            b[j] -= factor * b[i]
            b[j]%=p
            for k in range(n):
                A[k][j] -= factor * A[k][i]
                A[k][j]%=p

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i] * mod_inv(A[i][i],p)
        x[i]%=p
        for j in range(i - 1, -1, -1):
            b[j] -= A[i][j] * x[i]
            b[j]%=p


    return x

def find_y(p):
        k=natural_log(p)
        print(k)
        ex= k * natural_log(k)/2
        print(ex)
        return int(exp(ex**(0.5)))

def probabilistic_discrete_log(x,g,p):
    
    def find_k(y):
        k=0
        primes=[]
        for i in range(2,y+1):
            if(is_prime(i)):
                primes.append(i)
                k+=1
        return k, primes
    def find_e(m, primes):
        exps=[]
        for i in primes:
            e=0
            while m%i==0:
                m//=i
                e+=1
            exps.append(e)
        return exps
    y= find_y(p)
    q= p-1
    v=[]
    r_list=[]
    s_list=[]
    k,primes=find_k(y)
    for _ in range (0, k+1):
        flag= True
        while flag:
            r= random.randint(2,q-1)
            s= random.randint(2,q-1)
            beta=(pow(g,r,p)*pow(x,s,p))%p
            delta= random.randint(1, p-1)
            delta= pow(delta, q, p)
            m=(beta*delta)%p
            flag = not(is_smooth(m,y))
        v.append(find_e(m, primes))
        r_list.append(r)
        s_list.append(s)
    A= v[:-1]
    b=v[-1]
    c= gaussian_elimination(A, b,p)
    c.append(-1)
    r_sum=0
    for i in range (0,k):
        r_sum+=r_list[i]*c[i]
        r_sum%=q
    s_sum=0
    for i in range (0,k):
        s_sum+=s_list[i]*c[i]
        s_sum%=q
    if s_sum==0:
        raise ValueError("No solution")
    return (mod_inv(s_sum,p)*r_sum*(q-1))%q
        
    
import soc24mathlib

def probabilistic_factor(N):
    if(N<10**12):
        return soc24mathlib.factor(N)
    if(soc24mathlib.is_prime(N)):
        return [(N,1)]
    factors=[]
    for i in range (2,1000):
        if(soc24mathlib.is_prime(i)):
            count=0
            while(N%i==0):
                count+=1
                N//=i
            if(count>0):
                factors.append((i,count))
    if(N<10**12):
        factors.extend(soc24mathlib.factor(N))
        return factors
    
    def gauss(M):
        marks = [False] * len(M)
        for j in range(len(M[0])):
            for i in range(len(M)):
                if M[i][j] == 1:
                    marks[i] = True
                    for k in range(j):
                        if M[i][k] == 1:
                            for row in range(len(M)):
                                M[row][k] = (M[row][k] + M[row][j]) % 2
                    for k in range(j + 1, len(M[0])):
                        if M[i][k] == 1:
                            for row in range(len(M)):
                                M[row][k] = (M[row][k] + M[row][j]) % 2
                    break
        return (marks, M)

    def get_dep_cols(row):
        return [i for i in range(len(row)) if row[i] == 1]

    def row_add(new_row, current):
        return [current[i] ^ M[new_row][i] for i in range(len(M[new_row]))]

    def is_dependent(cols, row):
        return any(row[i] == 1 for i in cols)

    def find_linear_deps(row):
        ret = []
        dep_cols = get_dep_cols(M[row])
        current_rows = [row]
        current_sum = M[row][:]
        for i in range(len(M)):
            if i == row:
                continue
            if is_dependent(dep_cols, M[i]):
                current_rows.append(i)
                current_sum = row_add(i, current_sum)
                if sum(current_sum) == 0:
                    ret.append(current_rows[:])
        return ret

    def testdep(dep):
        x = y = 1
        for row in dep:
            x *= smooth_vals[row][0]
            y *= smooth_vals[row][1]
        return xgcd(x - soc24mathlib.floor_sqrt(y), N)[0]

    def xgcd(a, b):
        prevx, x = 1, 0
        prevy, y = 0, 1
        while b:
            q, r = divmod(a, b)
            x, prevx = prevx - q * x, x  
            y, prevy = prevy - q * y, y
            a, b = b, r
        return a, prevx, prevy

    def create_base(n, B):
        base = []
        i = 2
        while len(base) < B:
            if soc24mathlib.legendre_symbol(n, i) == 1:
                base.append(i)
            i += 1
            while not soc24mathlib.is_prime(i):
                i += 1
        return base

    def poly(x, a, b, n):
        return ((a * x + b) ** 2) - n

    def solve(a, b, n):
        start_vals = []
        for p in base:
            ainv=soc24mathlib.mod_inv(a, p)
            r1 = soc24mathlib.modular_sqrt_prime(n, p)
            r2 = (-1 * r1) % p
            start1 = (ainv * (r1 - b)) % p
            start2 = (ainv * (r2 - b)) % p
            start_vals.append([start1, start2])
        return start_vals

    def trial(n, base):
        ret = [0] * len(base)
        if n > 0:
            for i in range(len(base)):
                while n % base[i] == 0:
                    n //= base[i]
                    ret[i] = (ret[i] + 1) % 2
        return ret

    a = 1
    b = soc24mathlib.floor_sqrt(N) + 1
    bound = 50
    base = create_base(N, bound)
    needed = soc24mathlib.euler_phi(base[-1]) + 1

    sieve_start = 0
    sieve_stop = 0
    sieve_interval = 100000

    M = []
    smooth_vals = []
    start_vals = solve(a, b, N)
    seen = set([])

    while len(smooth_vals) < needed:
        sieve_start = sieve_stop
        sieve_stop += sieve_interval
        interval = [poly(x, a, b, N) for x in range(sieve_start, sieve_stop)]

        for p in range(len(base)):
            t = start_vals[p][0]
            while start_vals[p][0] < sieve_start + sieve_interval:
                while interval[start_vals[p][0] - sieve_start] % base[p] == 0:
                    interval[start_vals[p][0] - sieve_start] //= base[p]
                start_vals[p][0] += base[p]
            if start_vals[p][1] != t:
                while start_vals[p][1] < sieve_start + sieve_interval:
                    while interval[start_vals[p][1] - sieve_start] % base[p] == 0:
                        interval[start_vals[p][1] - sieve_start] //= base[p]
                    start_vals[p][1] += base[p]
        for i in range(sieve_interval):
            if interval[i] == 1:
                x = sieve_start + i
                y = poly(x, a, b, N)
                exp = trial(y, base)
                if not tuple(exp) in seen:
                    smooth_vals.append(((a * x) + b, y))
                    M.append(exp)
                    seen |= set([tuple(exp)])

    marks, M = gauss(M)
    

    for i in range(len(marks)):
        if not marks[i]:
            deps = find_linear_deps(i)
            for dep in deps:
                gcd = testdep(dep)
                if gcd != 1 and gcd != N:
                    print(gcd, N // gcd )
                    if gcd < 10**12:
                        factors.extend(soc24mathlib.factor(gcd))
                    else:
                        factors.extend(probabilistic_factor(gcd))
                    if (N // gcd) < 10**12:
                        factors.extend(soc24mathlib.factor(N // gcd))
                    else:
                        factors.extend(probabilistic_factor(N // gcd))
                    return factors




