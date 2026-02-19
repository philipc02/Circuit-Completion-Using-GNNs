spice
* SPICE netlist for the circuit

VCC 3 0 DC 10
Vi 9 0

R1 3 6 40k
R2 6 0 5k
RS 9 6 0.5k
RC 3 4 5k
RE 5 0 0.5k
RL 4 7 2.5k

C1 9 6 4.7u
C2 4 0 1e99
CE 5 0 1e99

Q1 4 6 5 QNPN

.model QNPN NPN(Is=1e-14 bf=100)
.END