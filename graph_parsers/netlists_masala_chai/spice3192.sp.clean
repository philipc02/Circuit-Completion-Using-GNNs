spice
* SPICE Netlist

* Voltage Source
VDD 5 0 DC 5V

* Current Source (Voltage Controlled Voltage Source)
EVOS 2 3 VALUE = {V(X) - V(Y)}

* MOSFETs
M3 4 2 1 1 NMOS
M4 5 2 4 4 NMOS
M5 5 5 4 4 PMOS

* Resistors
R1 2 3 1k
R2 3 0 1k
R3 2 0 1k
R4 5 4 1k

* BJTs
Q1 2 2 6 QNPN
Q2 3 3 0 QNPN

* Model Definitions (Assuming generic models for illustration)
.model NMOS NMOS
.model PMOS PMOS
.model QNPN NPN

.end