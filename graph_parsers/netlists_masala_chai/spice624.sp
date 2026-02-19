* SPICE Netlist for the given schematic

Vin IN 0 DC 0V
D1 IN X D
D2 X Y D
C1 X Y 1uF
Vb Y 0 DC 2V
R1 X OUT 1k
R2 OUT 0 1k

* Model for Diodes
.model D D

.end