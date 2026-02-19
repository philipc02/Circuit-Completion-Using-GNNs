* SPICE netlist

V1 2 6 DC 40
R1 2 5 120
RL 5 6 RL
D1 4 3 ZENER

* Model statement for Zener Diode
.model ZENER D (BV=VZ)

.END