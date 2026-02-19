plaintext
* SPICE netlist for the given schematic

VCC 8 0 DC 12
VIN 1 0 AC 1

R1 8 2 50
R2 2 3 100
RE 6 5 16
RL 4 7 16

C1 1 2 1u
C2 4 5 1u

Q1 4 2 6 NPN

*.MODEL NPN NPN(IS=1E-14 BF=100)

.END