plaintext
* Netlist for the given schematic

* Current Source
IREF 6 8 DC IREF

* Transistor MREF (PMOS)
MREF 8 8 2 2 PMOS W=W0

* Series NMOS Transistors
M1 8 2 3 3 NMOS W=W0
M2 2 2 3 3 NMOS W=W0
M3 2 2 2 2 NMOS W=W0
M4 4 2 2 2 NMOS W=W0

* Voltage Source
VDD 6 0 DC VDD

* End of netlist