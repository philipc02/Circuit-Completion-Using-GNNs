spice
* SPICE netlist for the given BJT circuit

Q1 2 4 5 NPN

RB 6 4 RB_value
RC 2 3 RC_value

VBB 6 7 DC VBB_value
VCC 3 7 DC VCC_value

* Define model for NPN transistor
.model NPN NPN (IS=1E-14 BF=100)

* End of netlist