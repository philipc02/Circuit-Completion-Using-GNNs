* SPICE Netlist

* Voltage Source
VDD 2 0 DC 1.8

* Current Source
IIN 6 3 DC 10u

* PMOS Transistors
M4 2 2 3 3 PMOS
M5 2 2 3 3 PMOS
M6 4 5 2 2 PMOS
M7 OUT1 5 2 2 PMOS
M8 2 3 4 4 PMOS
M9 2 4 5 5 PMOS

* NMOS Transistors
M1 3 6 6 6 NMOS
M2 3 3 6 6 NMOS
M3 3 3 6 6 NMOS
M10 5 5 3 3 NMOS
M11 OUT2 5 3 3 NMOS

* End of netlist