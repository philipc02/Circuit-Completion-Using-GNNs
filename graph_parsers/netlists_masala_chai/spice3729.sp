plaintext
* SPICE Netlist
VDD 3 7 DC 9V
ID 3 2 DC

M1 3 2 5 5 NMOS
M2 5 2 4 4 NMOS
M3 4 2 7 7 NMOS

*.model NMOS NMOS level=1
.end