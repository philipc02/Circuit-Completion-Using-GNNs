spice
* SPICE netlist for the provided schematic
M1 2 6 3 3 NMOS
RD 5 2 1k    * Assuming resistance value
RF 6 2 1k    * Assuming resistance value
VDD 4 5 DC 5V
Vin 6 0 DC 0V  * Assuming DC voltage source for input

* Model Definitions
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=120u)