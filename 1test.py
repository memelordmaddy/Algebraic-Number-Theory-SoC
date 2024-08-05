import soc24mathlib

assert soc24mathlib.pair_gcd(10, 5) == 5
assert soc24mathlib.pair_gcd(18, 81) == 9
assert soc24mathlib.pair_gcd(23005294688395892, 13345973765660648) == 4

assert soc24mathlib.pair_egcd(10, 5) == (0, 1, 5)
assert soc24mathlib.pair_egcd(18, 81) == (-4, 1, 9)
assert soc24mathlib.pair_egcd(23005294688395892, 13345973765660648) == (1108582724401689, -1910933792403058, 4)

assert soc24mathlib.gcd(9, 6, 27) == 3
assert soc24mathlib.gcd(256, 1024, 4096) == 256
assert soc24mathlib.gcd(13257, 17, 32892, 1328914) == 1

assert soc24mathlib.pair_lcm(10, 5) == 10
assert soc24mathlib.pair_lcm(18, 81) == 162
assert soc24mathlib.pair_lcm(23005294688395892, 13345973765660648) == 76757014845655956622768187314504

assert soc24mathlib.are_relatively_prime(10, 5) == False
assert soc24mathlib.are_relatively_prime(18, 81) == False
assert soc24mathlib.are_relatively_prime(54678052946438395891, 13345123132648) == True

assert soc24mathlib.mod_inv(3, 7) == 5
assert soc24mathlib.mod_inv(5, 13) == 8
assert soc24mathlib.mod_inv(2, 312345674079547151037918331725178312522478809653607352546657135738291654855733134069982077700935127515340479970913704499650782485828349263440468316632391) == 156172837039773575518959165862589156261239404826803676273328567869145827427866567034991038850467563757670239985456852249825391242914174631720234158316196
test_var = False
try:
    soc24mathlib.mod_inv(2, 6)
except:
    test_var = True
assert test_var == True

assert soc24mathlib.crt([3, 7, 8], [8, 13, 19]) == 787
assert soc24mathlib.crt([3137860500, 971430794, 971430794, 3777100181, 3309520681], [46864982125, 4287397436773, 5655291685133, 6223424679518, 256467105035203]) == 387964222001150943010209499163165460472906239773100251079584625

assert soc24mathlib.is_quadratic_residue_prime(3, 7) == -1
assert soc24mathlib.is_quadratic_residue_prime(9, 13) == 1
assert soc24mathlib.is_quadratic_residue_prime(0, 17) == 0
assert soc24mathlib.is_quadratic_residue_prime(36249236958, 312345674079547151037918331725178312522478809653607352546657135738291654855733134069982077700935127515340479970913704499650782485828349263440468316632391) == -1

assert soc24mathlib.is_quadratic_residue_prime_power(4, 7, 2) == 1
assert soc24mathlib.is_quadratic_residue_prime_power(52, 13, 3) == 0
assert soc24mathlib.is_quadratic_residue_prime_power(349, 1789, 1024) == -1

print("All tests passed!")
