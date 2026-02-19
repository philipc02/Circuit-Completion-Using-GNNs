spice
* SPICE Netlist

V1 5 6 DC 12V

R1 5 2 1k
R2 4 3 1k
R3 2 6 1k
R4 3 6 1k

* Analysis
.DC V1 0V 12V 1V
.END