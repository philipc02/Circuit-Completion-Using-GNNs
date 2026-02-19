spice
* SPICE Netlist for the circuit

V1 Vin 6 DC 5
RS 6 4 1k
RL 3 0 1k
Q1 3 4 2 QMOD
D1 2 5 DZ
VZ 5 0 DC 5.1

.model QMOD NPN
.model DZ D IS=1E-14 N=1 BV=5.1

.END