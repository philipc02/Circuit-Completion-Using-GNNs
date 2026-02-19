* SPICE Netlist for the given circuit

VDD VCC 0 DC 5
VSS VEE 0 DC -2.5
VBIAS VB 0 DC 2.5

* PMOS Transistor (Q1)
M1 V1 VB VCC VCC PMOS_MODEL

* NMOS Transistor (Q2)
M2 V2 0 N1 N1 NMOS_MODEL

* Resistor
R1 N1 VEE 1k

* Voltage Source connections
V1 VB 0 DC 2.5
V2 VCC 0 DC 5
V3 0 VEE DC 2.5

.model PMOS_MODEL PMOS
.model NMOS_MODEL NMOS

.end