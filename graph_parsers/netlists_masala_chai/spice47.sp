spice
* SPICE netlist for the given schematic

R1 5 6 ro

I1 7 3 (gm + gmb)*vsg
I2 2 3 (gm + gmb)*vsg

* Define connections
S 7 0
D 2 0
G 3 0

.END