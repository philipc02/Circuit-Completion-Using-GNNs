plaintext
* SPICE Netlist

Q1 5 2 3 nq1
Q2 2 4 0 nq2
I1 2 6 DC 1mA
I2 3 4 DC 1mA
VCC 3 0 DC 10V

.model nq1 NPN 
.model nq2 NPN 
.end