spice
* SPICE Netlist for the circuit

VCC 3 0 DC 2.5
Vx 2 0 DC 0

Ix 3 2 DC 0

R1 4 3 100k
R2 4 3 1k
R3 3 2 1k
R4 4 0 500

C1 3 0 1u
C2 4 0 1u

Q1 4 3 4 QMOD

.model QMOD NPN