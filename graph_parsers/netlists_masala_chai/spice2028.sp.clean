spice
* SPICE Netlist

V1 1 0 DC OFFSET
C1 1 2 C1_VALUE
C2 2 0 C2_VALUE
R1 3 0 R1_VALUE
R2 2 3 R2_VALUE
XOpamp 0 3 Vout OpampModel

*Model Definition
.subckt OpampModel 1 2 3
Rin 1 2 100MEG
Eout 3 0 1 2 1MEG
.ends OpampModel

.end