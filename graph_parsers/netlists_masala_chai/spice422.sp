spice
* Netlist for the provided schematic

* NMOS Transistors
M1 3 1 5 5 NMOS
M2 3 2 5 5 NMOS
M3 3 3 5 5 NMOS

* PMOS Transistors
M4 7 3 9 9 PMOS
M11 2 8 7 7 PMOS
M6 4 6 7 7 PMOS

* Current Sources
I1 7 9 DC
I2 8 6 DC
I3 7 10 DC

* Capacitor
C1 8 2 Value

* Voltage Source
V1 11 4 DC

* Nodes
* 1: in+
* 2: Vo
* 3: in-
* 4: -
* 5: VSS
* 6: 
* 7: VDD
* 8: VB
* 9: I1
* 10: I3
* 11: V1

.end