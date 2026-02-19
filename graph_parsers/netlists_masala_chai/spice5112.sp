plaintext
* SPICE Netlist

* NMOS Transistor
MN1 2 3 0 0 NMOS_MODEL

* Operational Amplifier
XOP1 1 2 2 OPAMP_MODEL

* Resistors
Rf 2 4 10k
R1 4 5 10k
R2 1 3 10k

* Voltage Sources
Vin 1 0 DC 1V

* Models
.model NMOS_MODEL NMOS (Level=1)
.model OPAMP_MODEL OPAMP

.end