spice
* NMOS Transistor
M1 3 4 2 2 NMOS

* Resistor
R1 1 3 R

* Voltage Sources
V1 4 0 V_i
V2 1 0 VDD_vdd
V3 0 2 VSS_vss

* Output
Vout 3 0 DC 0

* Model for NMOS (add specific parameters as needed)
.model NMOS NMOS (Level=1)