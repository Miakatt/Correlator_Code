#!/usr/bin/env python
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

X = 1920
Y = 1080
someX, someY = 0.5, 0.5
plt.figure()
currentAxis = plt.gca()
for i in range(10, Y, 50):
    currentAxis.add_patch(Rectangle((someX - i, someY - i), 2*i, 2*i, fill=None, alpha=1))
plt.xlim(-Y/2,Y/2)
plt.ylim(-X/2,X/2)
currentAxis.xaxis.set_visible(False)
currentAxis.yaxis.set_visible(False)
currentAxis.set_xticks([])
currentAxis.set_yticks([])

plt.savefig('Rectangle.png', format='png', dpi=1000, bbox='tight')
plt.show()
