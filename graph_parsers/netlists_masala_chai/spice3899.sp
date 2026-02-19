spice
* SPICE netlist for BJT circuit

VCC 5 0 DC 5
RB 4 3 RB_value
RC 5 2 RC_value
Q1 2 3 0 MyNPN

.model MyNPN NPN(Is=1e-16 BF=100 VAF=100)

.end