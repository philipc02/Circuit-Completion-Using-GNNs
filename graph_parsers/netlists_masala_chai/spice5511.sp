spice
* NPN BJT Differential Amplifier
Vcc 2 0 DC Vcc
Vin 4 0 DC Vin
Vee 0 4 DC Vee

R1 2 1 1k
R2 2 5 1k
R3 3 4 1k

Q1 3 4 1 NPN
Q2 5 4 3 NPN

.model NPN npn(IS=1e-15 BF=100)