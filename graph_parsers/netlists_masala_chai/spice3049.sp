plaintext
* SPICE Netlist for the Circuit
* NMOS Transistors
M1 X 2 6 6 NMOS
M2 Y 3 6 6 NMOS

* PMOS Transistors
M3 X 5 5 5 PMOS
M4 Y 5 5 5 PMOS

* Current Source
Iss 6 0 DC

* Voltage Source
VDD 5 0 DC

* Nodes:
* 2 = node X
* 3 = node Y
* 5 = VDD
* 6 = ground (connected to Iss)