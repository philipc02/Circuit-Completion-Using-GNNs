spice
* SPICE Netlist

* Transistor
Q1 3 6 7 NPN

* Resistors
R1 6 3 1k
R2 4 7 1k
RE 7 4 1k

* Capacitors
C1 3 4 10uF
C2 2 4 10uF
C3 4 6 10uF
CE 7 4 10uF

* Inductors
L_RF 3 5 10uH
L1 3 3 10uH

* Voltage Sources
VCC 6 4 DC 10V

* Connections
Vout 3

.MODEL NPN NPN (IS=1E-15 BF=100)
.END