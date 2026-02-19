spice
* SPICE Netlist
* Components
M1 6 5 3 3 NMOS
M2 5 7 3 3 NMOS
I1 2 6 DC * Current source I1
Iin 7 3 DC * Current source Iin
VDD 2 0 DC 5V * Voltage source VDD

* Model Definitions (assuming basic level models)
.model NMOS NMOS LEVEL=1
.end