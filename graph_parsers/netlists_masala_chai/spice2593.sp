spice
* SPICE Netlist for the given schematic

* Transistors
M1 2 2 2 2 NMOS
M2 3 2 3 3 NMOS

* Resistors
RD1 6 2 RD
RD2 6 3 RD
rO1 2 2 rO1
rO2 3 3 rO2

* Current Source
I1 2 4 ISS

* Voltage Source
V1 2 4 DC

* Node Definitions
* 2: Source of M1
* 3: Source of M2
* 6: VDD
* 4: Ground

.END