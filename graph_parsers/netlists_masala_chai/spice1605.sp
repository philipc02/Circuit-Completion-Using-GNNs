spice
* Transistors
M1 3 1 0 0 NMOS
M2 4 5 0 0 NMOS
M3 2 2 5 5 PMOS
M4 2 5 5 5 PMOS

* Current Sources
I1 2 0 DC
I2 2 5 DC
I3 0 3 DC
I4 0 4 DC

* Voltage Sources
V1 5 0 VCC
Vb 2 0 DC

* Resistor
R1 3 4 Rs

* Nodes
* 0 - Ground
* 1 - Vin1
* 2 - Vb
* 3 - Q1/Q3 Connection
* 4 - Q2/Q4 Connection
* 5 - Vout

.model NMOS NMOS(...)
.model PMOS PMOS(...)
.end