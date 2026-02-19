plaintext
* SPICE Netlist for the given schematic

* NMOS Transistors
M1 (3 Vin 3 3) NMOS
M2 (Vout 2 3 3) NMOS

* PMOS Transistors
M3 (Vout 2 VDD VDD) PMOS
M4 (2 Vout VDD VDD) PMOS

* Current Source
Iss 3 0 DC 1mA

* Resistor
R1 Vout 0 1k

* Define Nodes
Vin 1
Vout 2
VDD 3

* End of netlist