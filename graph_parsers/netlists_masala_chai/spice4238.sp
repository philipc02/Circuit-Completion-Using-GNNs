plaintext
* SPICE netlist for the given schematic

IREF 9 4 DC IREF

* PMOS Transistors
M1 3 7 3 3 PMOS
M2 3 5 3 3 PMOS

* NMOS Transistors
M3 4 7 3 3 NMOS
M4 2 8 3 3 NMOS

VDD 9 0 DC VDD
VSS 3 0 DC VSS

* Connections
* Node 2 is V_D4
* Node 6 is I_O