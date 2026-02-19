plaintext
* SPICE Netlist for the given schematic

Vcc 1 0 DC 5
Vs 4 0 AC 1

R1 1 2 1.2k
R2 5 0 1.2k
RS 4 5 300
RE 6 0 50
RL 3 0 10

C1 4 5 1u
C2 6 3 1u

Q1 2 4 6 NPN

.END