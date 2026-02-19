spice
* SPICE Netlist
V1 10 11 DC 1.5V
V2 9 2 DC -1.5V

* Current Sources
IREF 8 2 DC <Value>
I1 4 2 DC <Value>

* Resistor
R1 8 2 R

* Transistors
Q1 2 2 2 NMOS
Q2 7 2 2 NMOS
Q3 5 4 11 PMOS
Q4 3 2 2 NMOS
Q5 2 7 2 NMOS
Q6 11 5 10 PMOS
Q7 2 2 9 NMOS

.model NMOS NMOS
.model PMOS PMOS