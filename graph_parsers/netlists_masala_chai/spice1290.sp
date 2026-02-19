spice
* Netlist for the schematic

* Voltage Source
VDD VDD 0 DC 1.8V

* PMOS Transistor
M2 Vout VDD VDD VDD PMOS W=W2 L=0.18

* NMOS Transistor
M1 Vout Vin 0 0 NMOS W=10 L=0.18

* Models
.model PMOS PMOS
.model NMOS NMOS

.ends