plaintext
* SPICE Netlist
* Transistors
M1 2 6 8 8 NMOS
M2 3 4 5 5 NMOS
M3 2 6 9 9 PMOS
M4 3 4 9 9 PMOS

* Voltage Sources
VREF 2 0 DC <VREF_Value>
V1 2 0 DC <V_Value>

* Current Source
IBIAS 5 7 DC <IBIAS_Value>

* Input Sources
VIN1 6 0 DC <VIN1_Value>
VIN2 4 0 DC <VIN2_Value>

* Global GND
.model NMOS NMOS
.model PMOS PMOS