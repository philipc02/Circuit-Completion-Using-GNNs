spice
* Components
R1 2 0 10k
R2 3 4 10k
C1 3 2 1uF
V1 5 0 DC 5

* Op-Amp modeled as an ideal component
* Node connections: non-inverting input, inverting input, V+, Vout
XOPAMP 2 2 5 Vout Opamp

* Voltage source
VOS 2 0 DC 5

.ends

* Subcircuit for op-amp
.subckt Opamp noninv inv Vplus Vout
* Ideal op-amp assumption for SPICE
G1 Vout 0 VALUE = { V(noninv, inv) * 1.0e6 }
R1 Vout 0 1.0e6
.ends