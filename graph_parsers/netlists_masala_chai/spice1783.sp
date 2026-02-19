spice
* SPICE Netlist

* NMOS Transistors
M1 2 4 2 2 NMOS
M2 2 6 2 2 NMOS

* PMOS Transistors
M3 7 4 7 7 PMOS
M4 0 6 7 7 PMOS

* Current Source
I1 2 3 DC Iss

* Resistors
R1 0 2 R1
R2 2 3 R2

* Voltage Source
VDD 7 0 DC VDD

* Inputs
Vin 4 0 DC Vin

* Outputs
Vout 0 3

.END