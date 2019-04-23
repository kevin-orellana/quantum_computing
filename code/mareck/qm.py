import numpy as np

from qutip import *
from scipy import *

# functions and operators for QuTip

# return an array of N 2x2 identity matrices
def oplst(N): return [qeye(2) for n in range(N)]

# returns the tensor product of N 2x2 identity matrices
def I(N)     : return tensor(oplst(N))

# 
def Sx(N, n) : op = oplst(N); op[n] = sigmax(); return tensor(op)
def Sy(N, n) : op = oplst(N); op[n] = sigmay(); return tensor(op)
def Sz(N, n) : op = oplst(N); op[n] = sigmaz(); return tensor(op)

def Ising(h, J, N) :
    H  = -0.5*h*sum([Sz(N, n) for n in range(0, N)])
    H  -= 0.5*J*sum([Sx(N, n)*Sx(N, n+1) for n in range(0, N-1)])
    return H

def evolve(psi0, H, times, res, measurement, initial) :
    measurements = initial
    def meas(t, psi) :
        measurement(t, psi, measurements)
    mesolve(H, psi0, times, [], meas)
    return measurements

# functions and operators for IBM Q

def rx(circ, q, tau) : circ.u3(2.*tau, 3.*np.pi/2., np.pi/2., q)
def ry(circ, q, tau) : circ.u3(2.*tau, 0., 0., q)
def rz(circ, q, tau) : circ.u1(2.*tau, q)

def rxx(circ, q1, q2, tau) :
    circ.cx(q1, q2)
    rx(circ, q1, tau)
    circ.cx(q1, q2)

def isingStep(circ, q, tau, h, J) :
    # magnetic field adjustment
    for i in range(len(q)) :
        rz(circ, q[i], -.5*h*tau)
    # inter-particle interaction adjustment
    for i in range(len(q)-1) :
        rxx(circ, q[i], q[i+1], -.5*J*tau)

def isingTrotter(circ, qs, tau, steps, h, J) :
    t = np.float(tau)/np.float(steps)
    for step in range(steps) :
        isingStep(circ, qs, t, h, J)

def confs(counts, N) :
    _counts = {}
    for n in range(2**N) :
        config = [int(x) for x in bin(n)[2:]]
        for i in range(N - len(config)) :
            config = [0] + config
        config = ''.join(map(str, config))
        if config not in counts.keys() :
            _counts[config] = 0
        else :
            _counts[config] = counts[config]
    return _counts
