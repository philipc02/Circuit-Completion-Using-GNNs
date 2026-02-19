* SPICE Netlist for the Bipolar Transistor Circuit

VCC 7 0 DC VCC
VEE 3 0 DC VEE

Q1 5 6 4 NPN
Q2 5 2 4 NPN

RC1 7 5 RC
RC2 7 5 RC
RE 4 3 RE

VBE1 6 4 DC VBE
VBE2 2 4 DC VBE

.model NPN NPN (IS=1E-16 BF=100 VAF=200)
.ends