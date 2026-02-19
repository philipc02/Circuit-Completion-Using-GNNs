plaintext
* SPICE netlist for given schematic

V1 0 8 DC 25
Vz 7 10 DC 7.5

R1 8 1 1k
R2 4 8 1.5k
R3 11 9 1k
R4 11 12 1k
R5 12 3 1k

Q1 5 4 7 NPN
Q2 9 11 4 PNP

.model NPN NPN
.model PNP PNP

.end