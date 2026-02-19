spice
* SPICE Netlist
Vcc 2 0 DC 15V
Vsig1 5 0 DC Vsig/2
Vsig2 5 0 DC Vsig/2
I1 3 0 DC 1mA
R5k_1 5 7 5k
R5k_2 5 6 5k
RL 6 5 5k
Rc1 2 6 10k
Rc2 2 4 10k
Re1 3 2 150
Re2 3 4 150
Q1 6 7 3 QNPN
Q2 4 6 3 QNPN
.model QNPN npn
.end