spice
* SPICE Netlist for the schematic
* Node mapping is based on the annotated image

* Voltage Source
VDD 4 0 DC VDD

* PMOS Transistors
* PMOS Q2: Drain (4), Gate (3), Source (4), Body (4)
MPQ2 4 3 4 4 PMOS

* PMOS Q6: Drain (3), Gate (2), Source (3), Body (3)
MPQ6 3 2 3 3 PMOS

* NMOS Transistor
* NMOS Q5: Drain (3), Gate (2), Source (0), Body (0)
MNQ5 3 2 0 0 NMOS

* End of Netlist