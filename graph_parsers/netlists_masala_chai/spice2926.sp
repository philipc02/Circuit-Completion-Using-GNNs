spice
* SPICE netlist for the given schematic

* NMOS Transistor
M1 3 Vin 2 2 NMOS

* PMOS Transistor
M2 2 4 0 0 PMOS

* Resistors
RD 3 VDD RD_value
RS 2 0 RS_value

* Capacitors
C1 4 0 C1_value
C2 2 0 C2_value

* Voltage Source
VDD VDD 0 VDD_value

* Define model parameters for NMOS and PMOS (example values)
.model NMOS NMOS (level=1 Vto=0.7)
.model PMOS PMOS (level=1 Vto=-0.7)

.end