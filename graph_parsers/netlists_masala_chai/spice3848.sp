spice
* NPN BJT Circuit

Vplus 5 0 DC 3
Vminus 2 0 DC -3
RE 5 6 1k
RC 2 22 1k
Q1 3 3 6 QMOD

.model QMOD NPN(IS=1e-15 BF=100)

.END