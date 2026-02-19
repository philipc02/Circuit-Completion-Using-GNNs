spice
* SPICE Netlist for the given schematic
I1 2 6 DC 1mA
R1 6 3 1k
D1 5 3 DiodeModel
D2 3 4 DiodeModel
V1 2 7 DC 15V
* Define the diode model
.model DiodeModel D