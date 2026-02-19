spice
* SPICE Netlist for the provided schematic

M1 3 1 5 5 NMOS
M2 3 2 5 5 NMOS
M5 5 9 4 4 NMOS

M3 3 6 7 7 PMOS
M4 8 6 7 7 PMOS
M3A 3 3 6 6 PMOS
M4A 4 3 6 6 PMOS
M1A 3 1 3 3 PMOS
M2A 4 2 4 4 PMOS

VDD 7 0 DC 1.8
VBias 7 3 DC 1.2
VBB1 6 0 DC 1.5
VBB2 3 0 DC 1.2
Vcmc 9 0 DC 0.9
VSS 0 4 DC 0

* Connections
* Node 7: VDD, VBias
* Node 3: Gate of M3, M3A, M1A
* Node 6: Gate of M4
* Node 8: Drain of M4
* Node 1: V1 input for M1, M1A
* Node 2: V2 input for M2, M2A
* Node 5: Source of M1, M2, Drain/Body of M5
* Node 4: Source of M5, VSS

.ends