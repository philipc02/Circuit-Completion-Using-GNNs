spice
*MOSFET Circuit
M1 2 Vb 2 2 NMOS
M2 Vout Vin VDD VDD PMOS
RS 2 0 1000

*Voltage Sources
VDD VDD 0 DC 5
Vin Vin 0 DC 1
Vb Vb 0 DC 1.5

*Analysis
.TRAN 1n 10n
.END