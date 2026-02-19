plaintext
* SPICE Netlist for the given schematic

M1 8 6 11 11 NMOS
M2 2 8 3 3 NMOS
M3 9 7 3 3 NMOS
M4 8 2 10 10 PMOS
M5 2 4 10 10 PMOS
M6 7 5 10 10 PMOS

IIN 1 8 DC <value>
IBIAS1 9 7 DC <value>
IBIAS2 5 10 DC <value>

R1 3 11 <value>

VCC 10 0 DC <value>

* Model declarations
.model NMOS NMOS (KP=<value> VTO=<value>)
.model PMOS PMOS (KP=<value> VTO=<value>)

.end