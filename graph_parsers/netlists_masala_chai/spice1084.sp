plaintext
* NMOS Circuit SPICE Netlist

V1 1 0 DC 1.8V
R1 1 2 5k
M1 2 3 0 0 NMOS L=0.18u W=2u
V2 3 0 DC 1V

.model NMOS NMOS
.end