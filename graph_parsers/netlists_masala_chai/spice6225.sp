spice
* SPICE netlist for the given schematic

* NMOS Transistors
M1 2 2 3 3 NMOS
M2 0 1 3 3 NMOS
M3 6 4 2 2 NMOS

* PMOS Transistor
M4 5 6 7 7 PMOS

* Current Source
I1 1 2 DC IB

* Voltage Sources
VDD 4 0 DC VDD_value
VSS 0 7 DC VSS_value

* Resistor
R1 5 6 RL_value

* End of netlist