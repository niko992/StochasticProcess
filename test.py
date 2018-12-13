from functions import likelihood, G, p, u
import numpy as np

P = 8
M = 4
x = 0.5
xx = np.linspace(0, 0.5, M)
csi = [1, 4, 2, 3, 5, 5, 6, 0]

print(u(xx, csi))

print(G(csi, M))
print(likelihood(csi, M))


print(p(0, csi, M))
