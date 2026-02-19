spice
* SPICE Netlist for the given circuit

V_IN 4 0 AC 1 SIN(0 1 1k)
C1 4 1 0.1u
R1 1 5 10k
Q1 3 1 2 QMOD
C2 2 5 1u
RL 2 5 10k
L1 3 2 1u
C3 3 3 220p
VCC 3 0 DC 30

* Model for the BJT
.model QMOD NPN(IS=1e-14 BF=100)

*.end