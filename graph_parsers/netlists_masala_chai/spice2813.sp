plaintext
* SPICE Netlist for the given schematic

VDD 1 0 DC 5V
V1 Vb 0 DC 2.5V
Vin 4 0 DC 1V

* Transistors
M1 6 5 7 7 PMOS
M2 7 2 3 3 NMOS

* Current Sources
I1 1 7 DC 1mA
I2 3 0 DC 1mA

* Resistor
RS 4 3 50

* Voltage Sources
Vout 6 0

* Connections
VDD 1 0

* End of netlist