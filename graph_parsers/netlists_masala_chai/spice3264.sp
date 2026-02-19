spice
* Differential Amplifier Circuit
* Node Definitions:
* 1: VDD
* 2: Common source for PMOS (M3, M4)
* 3: Vout
* 4: Differential pair output
* 5: Common gate for PMOS
* 6: Vin (for M1)
* 7: Gate of M1 (Vin)
* 8: Gate of M2 (Vb)
* 9: Source of M2

M1 4 7 6 6 NMOS
M2 4 8 9 9 NMOS
M3 3 5 2 2 PMOS
M4 4 5 2 2 PMOS
I1 4 0 DC ISS

VDD 2 0 DC VDD
VIN 7 0 DC Vin
VB 8 0 DC Vb

.model NMOS NMOS
.model PMOS PMOS