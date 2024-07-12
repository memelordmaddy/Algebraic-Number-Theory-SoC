import soc24mathlib
import time
import random
n=611953
s1=0
for _ in range (1,10):
    start=time.time()
    i=random.randint(1,n-1)
    k=soc24mathlib.order(i,n)
    end=time.time()
    s1+= 1000*(end-start)

print("Avg Time taken: ",s1/10,"ms")
s2=0
for _ in range (1,10):
    start=time.time()
    i=random.randint(1,n-1)
    a=i
    for j in range (1,n):
        if(i==1):
            break
        i=(i*a)%n
    #assert pow(a,j,n)==1
    end=time.time()
    s2+=1000*(end-start)

print("Avg Time taken: ",s2/10,"ms")