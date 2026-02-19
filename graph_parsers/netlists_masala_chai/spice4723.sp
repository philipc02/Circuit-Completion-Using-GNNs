plaintext
* SPICE Netlist for the Given Circuit

VBB 4 0 DC 12V
VCC 3 0 DC 12V
RB 4 2 220k
RC 2 3 1k
Q1 2 2 4 BJT

.model BJT NPN (IS=1E-16 BF=100)

.end