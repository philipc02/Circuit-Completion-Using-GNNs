* Example SPICE Netlist

* Voltage Source
VDD 7 0 DC 5V

* Current Source
Iin 6 0 DC 1mA

* Resistors
RD 4 7 1k
RF 2 5 1k
RS 5 0 500

* NMOS Transistor
M1 3 6 6 6 NMOS

* PMOS Transistor
M2 7 3 5 5 PMOS

* Model Definitions
.model NMOS NMOS (LEVEL=1 VTO=1 KP=120u)
.model PMOS PMOS (LEVEL=1 VTO=-1 KP=50u)

*.end