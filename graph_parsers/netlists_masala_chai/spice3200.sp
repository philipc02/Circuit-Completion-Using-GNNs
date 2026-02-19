spice
* SPICE Netlist

* Voltage Source
VR1 0 2 DC 1.8

* Current Source
I_nA 9 0 DC 1e-6

* Resistors
R1 21 0 10k
R2 21 3 20k
R3 22 3 30k
R4 F 0 40k
R5 E Vout 50k
R6 2 0 60k
RM 0 9 70k

* NMOS Transistors
M1 21 3 0 0 NMOSmodel
M2 21 5 3 0 NMOSmodel
M3 22 11 10 0 NMOSmodel
M4 22 6 5 0 NMOSmodel
M5 10 3 11 0 NMOSmodel
M6 2 2 6 0 NMOSmodel
M7 2 7 7 0 NMOSmodel
M8 2 3 0 0 NMOSmodel
M9 Vout 4 2 0 NMOSmodel

* PMOS Transistors
M10 2 2 VDDL 2 PMOSmodel
M11 Vout 2 3 2 PMOSmodel

* Operational Amplifier
A1 E F Vout opampmodel

.model NMOSmodel NMOS (Level=1)
.model PMOSmodel PMOS (Level=1)
.model opampmodel opamp