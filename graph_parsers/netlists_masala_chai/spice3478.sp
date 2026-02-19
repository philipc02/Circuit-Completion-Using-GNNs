spice
* SPICE netlist for the given schematic

VDD 6 0 DC VDD
VSS 9 0 DC VSS

R0 6 3 R0

I1 3 8 DC (W1/LR)/(WR/LR)
I2 8 7 DC (W2/LR)/(WR/LR)
I3 7 4 DC (W3/LR)/(WR/LR)

M1 3 2 4 4 Qref
M2 8 5 2 2 Q1
M3 7 5 2 2 Q2
M4 4 5 2 2 Q3

* NMOS model - replace with appropriate model parameters
.model Qref NMOS
.model Q1 NMOS
.model Q2 NMOS
.model Q3 NMOS

.end