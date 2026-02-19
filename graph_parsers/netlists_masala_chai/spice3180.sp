spice
* SPICE Netlist for the provided schematic

* NMOS Transistors
M1 3 2 2 2 NMOS
M2 4 2 2 2 NMOS

* PMOS Transistors
M3 3 5 5 5 PMOS
M4 4 5 5 5 PMOS

* NPN BJTs
Q1 9 8 0 NPN
Q2 3 8 0 NPN
Q3 4 7 0 NPN
Q4 6 7 0 NPN

* Resistors
R1 2 7 R1_value
R2 4 2 R2_value

* Voltage Source
VDD 5 0 DC VDD_value

* End of netlist