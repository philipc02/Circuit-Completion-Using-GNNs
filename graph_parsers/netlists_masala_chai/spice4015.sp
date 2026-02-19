spice
* SPICE Netlist for the given circuit

VCC 9 0 DC 15
Vs 4 0 AC 1 SIN(0 1 1k)

R1 9 5 1k
R2 5 6 1k
RE 6 0 1k
Ro 1 7 1k
RL 7 0 1k

C1 4 10 10u
C2 2 7 10u

* NPN Transistor
Q1 3 5 6 QNPN

.model QNPN NPN (IS=1E-16 BF=100)

.end