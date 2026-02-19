spice
* SPICE netlist for the given schematic

V1 2 0 DC 2.5
Vin 1 0 AC 1V

R1 2 3 50k
R2 3 4 1k
R3 5 0 100

C1 1 3 0.5p
C2 4 5 0.5p

Q1 3 1 5 QNPN
Q2 4 3 6 QNPN

I1 6 0 DC 1mA

.model QNPN NPN(IS=1E-14 BF=100)

.end