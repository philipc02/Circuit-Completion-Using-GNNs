spice
* SPICE Netlist for the Given Circuit

Vi 8 0 DC 0V
RSi 8 3 1k
R1 7 2 10k
R2 9 3 5k
RC 2 6 4.7k
RE 4 0 560
CC 3 2 10uF
Q1 6 2 4 Q2N2222

VCC 7 0 DC 15V

.model Q2N2222 NPN (IS=1E-14 BF=200)
.END