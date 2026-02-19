spice
* SPICE Netlist for the given schematic

V1 3 0 DC 5V
V2 4 0 DC -5V

R1 3 3 Q2C 1k
R2 4 B 10k
R3 Q2E 2 1k

Q2 Q2C B Q2E NPN

.end