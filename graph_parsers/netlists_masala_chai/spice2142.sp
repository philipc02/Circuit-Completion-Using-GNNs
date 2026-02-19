* SPICE Netlist for the given schematic

VDD 7 3 DC 1.8V

* Resistors
R1 7 3 1k
RD 7 2 1k
R2 3 5 1k

* Capacitors
C1 8 3 1u
C2 2 4 1u

* NMOS Transistor
M1 2 3 2 2 NMOS_MODEL

* NMOS Model
.model NMOS_MODEL NMOS (LEVEL=1)

* .end