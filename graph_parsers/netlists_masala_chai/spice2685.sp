spice
* SPICE netlist for the schematic

VDD 4 0 DC VDD_value
ro3 5 6 ro3_value
ro4 4 2 ro4_value
M3 4 3 6 6 PMOS L=1u W=1u
M4 2 3 4 4 PMOS L=1u W=1u
I1 6 0 DC delta_I1_value
I2 2 0 DC delta_I2_value

* Connect Vout to node 2
Vout 2 0

* Include model parameters
.model PMOS PMOS LEVEL=1 VTO=0.7 KP=50u

.end