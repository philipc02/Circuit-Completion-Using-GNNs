* NMOS LED Driver Circuit

V1 VI 0 DC
V2 5V 0 DC 5

M1 5 3 2 2 NMOS
R1 5 5V RD
D1 2 0 LED

.model NMOS NMOS
.model LED D

.end