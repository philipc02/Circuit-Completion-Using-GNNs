plaintext
* SPICE Netlist for the given circuit

Vplus 7 0 DC VPLUS_DC
Vminus 3 0 DC VMINUS_DC
Vin 2 0 DC VIN_DC

R1 7 4 R1_VALUE
R2 4 3 R2_VALUE
RC 7 6 RC_VALUE
RE 8 3 RE_VALUE
RL 6 1 RL_VALUE
RS 1 2 RS_VALUE

CB 5 4 CB_VALUE
CC 8 6 CC_VALUE
CL 1 0 CL_VALUE

Q1 6 4 8 QNPN_MODEL

* Define the model for the NPN transistor
.model QNPN_MODEL NPN(IS=1E-16 BF=100)

* End of netlist