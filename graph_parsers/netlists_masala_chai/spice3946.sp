plaintext
* SPICE Netlist
Vcc 7 0 DC 5V
Vs 6 0 SIN(0 1 1k)
Rs 6 2 0.5k
R1 7 2 1k
R2 2 3 1k
Rc 7 5 1k
Re 5 3 1k
Cc 2 4 10u
Q1 5 4 3 QNPN
.model QNPN NPN

.end