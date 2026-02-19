plaintext
* SPICE Netlist
Vsig 5 4 DC 0
R1 4 2 500k
C1 2 1 1u
RG 1 6 10Meg
Q1 6 2 5 NPN
R2 5 0 6.8k
Q2 9 5 0 NPN
R3 9 0 3k
C2 9 3 1u
R4 3 0 1k
Vcc 9 0 5

*.model declaration for NPN transistors
.model NPN NPN (IS=1E-14 BF=100)

.end