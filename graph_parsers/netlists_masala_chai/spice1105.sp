plaintext
* SPICE Netlist
*MOSFET Definition: Default
.model NMOS NMOS (LEVEL=1)
.model PMOS PMOS (LEVEL=1)

* Voltage Source
VB 2 0 DC VALUE

* Current Sources
ID1 4 2 DC VALUE
ID2 3 2 DC VALUE

* Transistors
M1 4 VB 2 2 NMOS
M2 3 4 2 2 NMOS

* End of Netlist