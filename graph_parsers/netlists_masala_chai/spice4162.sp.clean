plaintext
* SPICE netlist for the given schematic

* Voltage Sources
V1 5 0 DC 5V
V2 9 0 DC -5V

* Resistors
R1 1 6 1k
RL 5 0 1k

* NMOS Transistors
M1 5 1 4 4 NMOS
M2 5 2 7 7 NMOS
M3 4 3 8 8 NMOS

* Node Definitions
* 1: Input node (gate of M1)
* 2: Intermediate node (gate of M2)
* 3: Intermediate node (gate of M3)
* 4: Source of M1, drain of M3, drain of M2
* 5: Output node (vo)
* 6: Connected to resistor R
* 7: Source of M2
* 8: Source of M3, connected to ground via resistor R
* 9: V2 (-5V) supply

.model NMOS NMOS
.end