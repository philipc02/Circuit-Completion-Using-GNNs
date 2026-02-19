plaintext
* SPICE Netlist for the given schematic

* NPN BJT Qn
Qn 4 2 0 NPN

* PNP BJT Qp
Qp 9 2 7 PNP

* Current Source
I_Bias 3 4 DC

* Resistors
R1 2 5 R1_value
R2 5 0 R2_value
RL 8 7 RL_value

* Capacitor
C1 2 5 C1_value

* Voltage Source
Vin 1 0 AC

* Connections
V+ 3 0 DC
V- 7 0 DC
Vo 9 8 DC

.model NPN npn
.model PNP pnp