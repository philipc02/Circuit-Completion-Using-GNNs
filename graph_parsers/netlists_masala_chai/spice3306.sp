spice
* SPICE Netlist

VDD 5 0 DC 5
Vcont 6 0 DC 5

* Transistors
M1 4 2 3 3 NMOS
M2 4 2 3 3 NMOS
M3 1 2 0 0 NMOS
M4 1 2 0 0 NMOS
M5 4 2 3 3 PMOS
M6 4 2 3 3 PMOS
M7 6 2 3B 3B NMOS
M8 6 2 3B 3B NMOS

* Resistors
R1 5 2 10k
R2 2 1 10k

* Current Source
I_SS 3B 0 DC 1mA

* End of Netlist