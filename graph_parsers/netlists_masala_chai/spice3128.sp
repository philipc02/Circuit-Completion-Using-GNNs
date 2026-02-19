plaintext
*MOSFETs
* NMOS transistors are denoted with model nmos, PMOS with pmos.
M1  1  2  26  26  nmos
M2  2  8  26  26  nmos
M3  27 11  3  3   nmos
M4  Vout  4  3  3  nmos
M5  4  32  23 23  nmos
M6  Vout  33 23 23 nmos
M9  3  27  7  7   nmos
M10 Vb4  28  7  7  nmos

M7  22  1  2  2   pmos
M8  Vdd 1  33 33  pmos

*Current Sources
I1  Vdd 4  DC 0
I2  Vdd 8  DC 0

*Voltage Sources
Vdd 33 0  DC VDD
Vb4  7  0  DC VB4