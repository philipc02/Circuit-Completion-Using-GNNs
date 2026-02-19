* Netlist for schematic

VDD 3 0 DC VDD
VB 4 0 DC VB
VR 8 0 DC VR

* MOSFETs
M1 3 4 0 0 Q1_NMOS
M2 2 6 0 0 Q2_NMOS
M3 5 3 3 5 Q3_PMOS
M4 2 5 5 5 Q4_PMOS
M5 7 8 0 0 Q5_NMOS

* Capacitor
C1 2 0 C_VALUE

* Current Source
I1 7 0 DC I_VALUE

.model Q1_NMOS NMOS
.model Q2_NMOS NMOS
.model Q3_PMOS PMOS
.model Q4_PMOS PMOS
.model Q5_NMOS NMOS

* End of netlist