* SPICE Netlist for the given schematic
* Transistor Q1 
Q1 Net2 Vb1 Net1 NPN

* Transistor Q2
Q2 Net1 0 0 NPN

* Resistors
RP Net2 Net1 RP_value
Rout Net2 0 Rout_value

* Voltage Source
Vb1 Vb1 0 DC Vb1_value

* End of Netlist