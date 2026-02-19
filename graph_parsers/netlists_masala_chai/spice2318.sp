* SPICE Netlist for Schematic

* NMOS Transistor
M1 3 IN 4 4 NMOS_Model

* Diode
D1 4 OUT Diode_Model

* Current Source
I1 4 5 DC IBIAS

* Models (example)
.model NMOS_Model NMOS
.model Diode_Model D