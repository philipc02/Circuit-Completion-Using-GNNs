spice
* SPICE netlist for the given schematic

* Voltage Source
V1 3 0 DC VDD

* Current Source
I1 7 0 DC ISS

* MOSFETs
M1 4 8 7 7 NMOS
M2 6 4 7 7 NMOS
M3 2 5 6 6 PMOS

* Resistor
R1 6 0 R1

* Node Assignments
* 1: Ground
* 2: VDD
* 3: Connection at VDD
* 4: Drain of M1 and gate of M2
* 5: Connection between M3 and I_out
* 6: Drain of M2, source of M3, and one terminal of R1
* 7: Iss connecting to ground
* 8: Gate of M1 (Vin)

.end