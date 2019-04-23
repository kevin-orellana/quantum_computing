import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from qutip import *
from scipy import *

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qm import Ising, evolve, isingTrotter, confs

from plotting import plot

def spinket(conf) :
    psi_list = [basis(2,n) for n in conf]
    return tensor(psi_list)

# input parameters
N = 4
t_max = 2.*np.pi
res = 60
times = np.linspace(0.0, t_max, res)

h = np.random.uniform(-1.,1.) # transverse field
J = np.random.uniform(-1.,1.) # coupling strength

# at which (discrete) evolution time steps we wish to compare QuTip and IBM Q
timesteps = []

# continuous time evolution using QuTip
H = Ising(h, J, N)

# prepare initial state and measured states
psi0 = tensor([basis(2, 0) for i in range(N)])

# define a callback function which will perform measurement for us
def measurement(t, psi, previous) :
    i = previous[0]
    previous[0] += 1
    for n in range(2**N) :
        conf = [int(x) for x in bin(n)[2:]]
        for j in range(N - len(conf)) :
            conf = [0] + conf
        psin = spinket(conf)
        previous[n+1][i] = np.power(np.abs((psin.dag()*psi).full()[0][0]), 2.)
    #previous[2][i] = np.power(np.abs((psib.dag()*psi).full()[0][0]), 2.)
initial = [0] + [np.zeros(res) for i in range(2**N)]

# perform the time-evolution using Qutip
results = evolve(psi0, H, times, res, measurement, initial)

# IBM Q
for measure_step in [int(floor(i))for i in linspace(0., res, 3, endpoint=False)] :
    tau = times[measure_step]
    shots = 2048
    q = QuantumRegister(4, 'q')
    c = ClassicalRegister(4, 'c')
    circ = QuantumCircuit(q)
    isingTrotter(circ, q, tau, 800, h, J)
    meas = QuantumCircuit(q, c)
    meas.barrier(q)
    meas.measure(q,c)
    qc = circ+meas
    job = execute(qc, backend = Aer.get_backend('qasm_simulator'), shots=shots, seed=8)
    result = job.result()
    counts = result.get_counts(qc)
    counts = confs(counts, N)
    print('\ntime step: ' + str(measure_step))
    print('{0:7} {1:7} {2:7}'.format('', 'IBM-Q', 'QuTip'))
    for i in range(2**N) :
        conf = [int(x) for x in bin(i)[2:]]
        for j in range(N - len(conf)) :
            conf = [0] + conf
        txt = ''.join(map(str, conf))
        print('{0:7} {1:7} {2:7}'.format('|'+txt+'>', str(np.round(np.float(counts[txt])/np.float(shots), 3)), np.round(results[i+1][measure_step], 3)))

'''
q = QuantumRegister(4, 'q')
c = ClassicalRegister(4, 'c')
circ = QuantumCircuit(q)
isingTrotter(circ, q, t_max, 1, h, J)
meas = QuantumCircuit(q, c)
meas.barrier(q)
meas.measure(q, c)
qc = circ+meas
drawing = qc.draw(output='mpl')
drawing.savefig('res_circuit.png')
'''

'''
# plot the results from continuous time evolution
fig, axes = plt.subplots()

ticks = np.linspace(0., t_max, 10)
ticks_labels = ['$'+str(round(t/np.pi, 2)) + '\pi$' for t in ticks]

ys = [psi0s, psi1s]
labels = ['$<%s|\psi>$' % txt_a, '$<%s|\psi>$' % txt_b]
colors = ['C0', 'C1']
plot_data = [times, times], ys, labels, colors
plot(axes, '$|%s>, |%s>$ states' % (txt_a, txt_b), plot_data, ticks=ticks, tlabels=ticks_labels)

fig.tight_layout()
#fig.savefig('res_sanity.png')
plt.show()
'''
