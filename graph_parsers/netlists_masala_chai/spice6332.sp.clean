spice
* SPICE Netlist for the provided schematic

*.model statements for PMOS and NMOS transistors
.model PMOS PMOS
.model NMOS NMOS

* Voltage Source
VDD 5 0 DC <VDD_value>

* PMOS Transistors
M1 7 5 5 5 PMOS
M2 8 7 7 7 PMOS

* NMOS Transistors
M3 7 2 3 3 NMOS
M4 8 4 3 3 NMOS

* Nodes
* 0 - Ground
* 2 - AO (Input)
* 3 - Ground
* 4 - BO (Input)
* 5 - VDD (Power supply, node marked as 5 and source for PMOS)
* 7 - X (Intermediate Node)
* 8 - Y (Output Node)

.END