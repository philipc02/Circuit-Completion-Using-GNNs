spice
* SPICE Netlist

* Voltage Source
VDD 5 0 DC 1.8V

* Current Sources
I1 3 0 DC 1mA
Iin 3 0

* NMOS Transistors
M1 2 3 0 0 NMOS
M2 2 4 5 5 NMOS
M3 5 4 0 0 NMOS

* Resistor
RF 2 3 RF_value

* .Model statement for NMOS
.model NMOS NMOS

.end