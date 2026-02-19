spice
* NMOS: drain gate source
M1 6 4 3 NMOS ; Q1
M2 5 1 2 NMOS ; Q2

* PMOS: drain gate source
M3 3 8 2 PMOS ; QS

* Resistors
R1 7 6 RD
R2 7 5 RD

* Voltage Sources
VDD 7 0 DC VDD_value
VSS 0 3 DC VSS_value
VBIAS 8 0 DC VBIAS_value