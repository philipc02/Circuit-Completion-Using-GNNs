* BJT Differential Pair with Resistor

Q1 2 1 3 NPN
Q2 2 3 0 NPN

RP 2 4 1k

Vb1 1 0 DC 1V
Vb2 2 0 DC 1V
Vout 4 0 DC 5V

.model NPN NPN(IS=1E-14 BF=100)

.end