* SPICE Netlist for the Given Circuit

* Voltage Sources
VCC 1 0 DC VCC
VEE 6 0 DC VEE

* Resistor
R1 1 2 R1_value

* Transistors
Q1 3 2 1 QNPN
Q2 3 4 6 QNPN
Q3 4 4 3 QNPN
Q4 5 3 6 QNPN

* Current Sources
IBIAS1 3 0 DC IBIAS1_value
IBIAS2 5 0 DC IBIAS2_value

* Models
.model QNPN NPN (IS=1E-14 BF=100)  ; Define NPN transistor model