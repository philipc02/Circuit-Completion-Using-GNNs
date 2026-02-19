spice
* SPICE Netlist for the given schematic

V1 7 0 DC 0
V2 3 0 DC 0
V3 5 0 DC 0
V4 2 0 DC 0

R1 7 3 1k
R2 3 2 1k
Rf 8 4 1k
R3 2 5 1k
R4 5 2 1k
R5 2 8 1k

XOP 2 2 4 opamp

* Define the subcircuit model for the opamp
.subckt opamp in- in+ out
* Add the specific properties of the opamp here
.ends opamp

.END