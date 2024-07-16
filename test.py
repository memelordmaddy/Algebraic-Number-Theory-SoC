import soc24mathlib

assert soc24mathlib.discrete_log(11, 2, 13) == 7
assert soc24mathlib.discrete_log(12384, 89, 3698849471) == 1261856717 

assert soc24mathlib.legendre_symbol(3, 7) == -1
assert soc24mathlib.legendre_symbol(9, 13) == 1
assert soc24mathlib.legendre_symbol(0, 17) == 0
assert soc24mathlib.legendre_symbol(36249236958, 312345674079547151037918331725178312522478809653607352546657135738291654855733134069982077700935127515340479970913704499650782485828349263440468316632391) == -1

assert soc24mathlib.jacobi_symbol(3, 79) == -1
assert soc24mathlib.jacobi_symbol(1789,3189045) ==-1
assert soc24mathlib.jacobi_symbol(7921, 489303) == 1
assert soc24mathlib.jacobi_symbol(136, 153) == 0

assert soc24mathlib.modular_sqrt_prime(11, 19) == 7
assert soc24mathlib.modular_sqrt_prime(12378, 3698849471) == 2301481823

#assert soc24mathlib.modular_sqrt_prime_power(11, 19, 8) == 2684202706
#assert soc24mathlib.modular_sqrt_prime_power(12378, 3698849471, 3) == 19725977363156848933505792157

#assert soc24mathlib.modular_sqrt(10, 15) == 5
#assert soc24mathlib.modular_sqrt(91, 157482) == 62855

assert soc24mathlib.is_smooth(1759590, 20) == True
assert soc24mathlib.is_smooth(906486, 150) == False

'''
assert soc24mathlib.probabilistic_discrete_log(11, 2, 13) == 7
assert soc24mathlib.probabilistic_discrete_log(12384, 89, 3698849471) == 1261856717
assert soc24mathlib.probabilistic_discrete_log(131313, 13, 17077114927) == 12294541275
'''
# Note that the first components of the tuples are the actual factors, while the second components are the multiplicities.
# The actual factors must be sorted in ascending order.

'''
assert soc24mathlib.probabilistic_factor(1) == []
assert soc24mathlib.probabilistic_factor(7) == [(7, 1)]
assert soc24mathlib.probabilistic_factor(243) == [(3, 5)]
assert soc24mathlib.probabilistic_factor(4104) == [(2, 3), (3, 3), (19, 1)]
assert soc24mathlib.probabilistic_factor(1408198281) == [(3, 1), (7, 1), (17, 1), (19, 1), (31, 1), (37, 1), (181, 1)]
'''
print("All tests passed!")