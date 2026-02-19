* Example SPICE netlist

C1 6 5 C1
C2 6 2 C2
R1 5 3 R1
R2 6 0 R2
RF1 3 2 RF1
RF2 2 0 RF2
RF3 4 0 RF3
D1 3 4 D1
D2 4 3 D2
XOP 3 2 Vout OPAMP_MODEL

* Ground
0 GND

* Control the parameters of the elements
.model D1 D(Is=1e-14)
.model D2 D(Is=1e-14)
.model OPAMP_MODEL OPAMP

.END