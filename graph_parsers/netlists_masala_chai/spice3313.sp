spice
* SPICE Netlist
VDD 2 0 DC 5V
Vin 5 0 DC 0V

* Transistors
MPMOS 3 3 6 6 PMOS L=1u W=10u
MNMOS 3 3 0 0 NMOS L=1u W=10u

* Current Source
I1 3 0 DC 1mA

* Inductors
L1 2 3 10uH
L2 3 3 10uH

* Capacitor
C1 3 0 1uF

* Resistor
R1 5 3 1k

* Diodes
D1 3 3 D
D2 3 8 D

.model PMOS PMOS
.model NMOS NMOS
.model D D

.end