plaintext
* Differential Amplifier SPICE Netlist

*.model statements for transistors
.model NMOS NMOS
.model PMOS PMOS

* NMOS Transistors
M1 5 1 6 6 NMOS
M2 2 2 6 6 NMOS
M3 6 3 6 6 NMOS
M4 2 2 6 6 NMOS

* PMOS Transistors
M5 7 3 4 4 PMOS
M6 7 2 4 4 PMOS

* Current Source
I1 6 0 DC IQ

* Voltage connections
V+ 7
V- 0

* Inputs
v1 1
v2 2

* Output
vo 3