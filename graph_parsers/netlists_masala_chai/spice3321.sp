spice
* NMOS amplifier circuit

M1 out in 0 0 NMOS
R1 3 out 2k
C1 out 2 10n
C2 3 out 10n
VDD 3 0 DC 10V
Vin in 0 DC 0V AC 1V

.tran 10n 1u
.ac dec 100 1 1Meg
.end