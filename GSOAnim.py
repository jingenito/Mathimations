import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

MAX = 5
MIN = -5

def GramSchmidt(matrix : np.array) -> np.array :
    result = np.zeros(matrix.shape)
    result[0] = matrix[0]
    for i in range(1,matrix.shape[0]) :
        sum = np.zeros((1,matrix.shape[1]))
        for j in range(i) :
            gso_coeff = np.dot(matrix[i],result[j]) / np.dot(result[j],result[j])
            sum += gso_coeff * result[j]
        result[i] = matrix[i] - sum
    return result

matrix = (MAX - MIN) * np.random.rand(3,3) + MIN
fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
ax.set_xlim(MIN, MAX)
ax.set_ylim(MIN, MAX)
ax.set_zlim(MIN, MAX)

origin = np.zeros(matrix.shape)
result = GramSchmidt(matrix)

#plot the original state in blue
quiver_init = ax.quiver(*origin, matrix[:,0], matrix[:,1], matrix[:,2], color=['b'], linewidths=2)
plt.pause(2)
#create a different quiver in red for the animation
quiver = ax.quiver(*origin, matrix[:,0], matrix[:,1], matrix[:,2], color=['r'], linewidths=2)

#animate function
def UpdateVectors(t) :
    global quiver
    quiver.remove()
    arr = (1-t)*matrix + t*result
    quiver = ax.quiver(*origin, arr[:,0], arr[:,1], arr[:,2], color=['r'])

#plot the final state
quiver_final = ax.quiver(*origin, result[:,0], result[:,1], result[:,2], color=['g'])
plt.pause(1)

filename = 'GSO_animation_random.gif'
anim = FuncAnimation(fig, UpdateVectors, frames=np.linspace(0,1,200), interval=20)
print('Saving as gif to ' + filename)
anim.save(filename, writer=PillowWriter(fps=30))
plt.show()