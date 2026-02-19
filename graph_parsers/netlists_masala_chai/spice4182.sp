plaintext
* SPICE Netlist for the given circuit

* Voltage Source
VS 1 0 DC 0

* Resistors
R1 4 3 1k
R2 3 2 1k
RF 2 3 1k
R3 1 3 1k
R4 3 0 1k

* Diodes
D1 3 3 DMOD
VZ 3 0 ZMOD

* Models
.model DMOD D
.model ZMOD Zener

* End of Netlist