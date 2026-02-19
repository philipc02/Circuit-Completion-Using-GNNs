spice
* SPICE Netlist for the given Schematic

* Voltage Sources
V1 3 2 DC 10V
V2 5 0 DC 25V

* PNP BJT
Q1 6 3 5 PNP

* Resistor
R1 6 4 R

* Ground
V0 4 0 DC 0V

* Note: Node 2 is used for the common connection between the voltage source V1 and the non-inverting terminal of the op-amp.