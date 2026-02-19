spice
* NMOS Differential Pair Circuit

VDD 5 0 DC 2.5V
VSS 6 0 DC -2.5V
VG1 1 0 DC
VG2 5 0 DC

I1 3 0 DC 0.5mA

M1 3 1 2 2 NMOS
M2 3 5 4 4 NMOS

R1 2 6 4k
R2 4 7 4k

.tran 1n
.end