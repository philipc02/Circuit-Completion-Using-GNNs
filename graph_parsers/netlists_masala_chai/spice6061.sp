* SPICE Netlist for the given schematic

V1 5 0 DC 0
Rc 4 3 1k
Q1 4 5 2 NPN_MODEL
Q2 3 2 22 NPN_MODEL
I1 2 0 DC 1mA

.model NPN_MODEL NPN

.end