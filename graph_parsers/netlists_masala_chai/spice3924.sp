plaintext
* SPICE Netlist for the given schematic

V1 8 6 DC 9V

R1 7 8 80k
R2 5 6 40k
R3 8 4 2k
R4 5 2 2k
R5 3 4 100
R6 4 6 200

* Assuming Q1 and Q2 are NPN transistors
Q1 5 8 2 NPN
Q2 4 3 6 NPN

.model NPN NPN(IS=1E-14 BF=100)

.end