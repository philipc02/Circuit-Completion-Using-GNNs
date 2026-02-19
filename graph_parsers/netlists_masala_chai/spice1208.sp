plaintext
*MOSFET Model Parameters
.model NMOS NMOS
.model PMOS PMOS

*Voltage Sources
VDD 5 0 DC VDD
Vb 3 0 DC Vb
Vin 1 0 DC Vin

*Transistors
M1 X 1 2 2 NMOS
M2 2 0 0 0 NMOS
M3 4 3 4 5 PMOS
M4 4 3 5 5 PMOS

*Outputs
*Vout1 at Node X
*Vout2 at Node Y