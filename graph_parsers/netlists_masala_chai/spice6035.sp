spice
* SPICE Netlist for the Given Circuit

* NMOS (Drain, Gate, Source)
M1 4 3 2 NMOS
M2 5 1 2 NMOS

* PMOS (Drain, Gate, Source)
M3 2 6 2 PMOS
M4 5 7 2 PMOS

* Resistors
R_ro3 6 8 ro3
R_gm3 4 4 1/gm3
R_ro1 4 3 ro1
R_RO2 3 2 RO2
R_ro4 5 2 ro4
R_1_gm4 5 2 1/gm4
R_Rin 8 2 Rin

* Voltage Source
Vx 2 1 Vx

* Ground
VSS 2 0 DC 0

.end