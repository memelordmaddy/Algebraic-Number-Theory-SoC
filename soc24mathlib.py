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
