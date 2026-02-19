plaintext
* SPICE Netlist for the Given Circuit

Vpi 2 0 DC 0
Q1 4 5 2 QNPN

R1 5 3 R
C1 3 4 C1
C2 3 2 C2
L1 4 6 L
Vo 6 0 DC 0

.model QNPN NPN (IS=1e-14 BF=200)

* End of netlist