plaintext
* SPICE Netlist for the Given Circuit

* Q1 NPN Transistor
Q1 4 10 9 NPN

* Q2 NPN Transistor
Q2 2 8 6 NPN

* Current Source I
I1 2 7 DC I

* Current Source 2I
I2 9 5 DC 2I

* Resistor Rin
Rin 10 9 Rin

* Voltage Nodes
VCC 7 0 DC VCC
VEE 0 5 DC VEE

* Output
Vout 3 6 DC 0

* Define models
.model NPN NPN