spice
* SPICE Netlist
Vi 1 2 DC 0
Rin 2 2 1k
Cc 1 2 10n

M1 5 6 4 4 NMOS
M2 6 5 7 7 NMOS

R1 3 4 1k
R2 4 6 1k
RS1 5 3 1k
RD1 5 6 1k
RS2 7 6 1k
Ro 7 6 1k

V+ 3 0 DC 1.8V
V- 6 0 DC -1.8V

.options post=2
.end