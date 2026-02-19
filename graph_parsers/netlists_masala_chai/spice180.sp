* SPICE Netlist for the given BJT circuit

VCC 1 0 DC 15V

R1 1 7 10k
R2 4 3 1k

* NPN Transistors
Q1 4 7 8 QNPN
Q2 2 4 3 QNPN

Iout 2 0 0

.model QNPN NPN (IS=1e-14 BF=100)

.end