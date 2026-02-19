spice
* Multi-stage amplifier circuit

V1 N1 0 DC Vin
Rs N1 N2 R_s
C1 N2 N3  C_1

* Stage 1
Q1 N6 N5 N4 N4 QNPN
Rc1 N5 Vcc R_C
Rl1 N6 0 R_L
R1 Vcc N5 R_1
R2 N5 0 R_2
Re N4 0 R_E

C2 N6 N7  C_2

* Stage 2
Q2 N10 N9 N8 N8 QNPN
Rc2 N9 Vcc R_C
Rl2 N10 0 R_L
R3 Vcc N9 R_1
R4 N9 0 R_2
Re2 N8 0 R_E

C3 N10 Vout C_3

* Load
Rl Vout 0 R_L

.model QNPN NPN(IS=1E-14 BF=100)
.end