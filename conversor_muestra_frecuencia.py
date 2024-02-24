fs = 44100
longitud = 32768
m1 = 20008
m2 = 2000 + (32768/4)
f1 = (fs * m1) / (2 * longitud)
f2 = (fs * m2) / (2 * longitud)
print(f1)
print(f2)
