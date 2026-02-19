plaintext
* SPICE Netlist for the Current Mirror

VDD 5 0 DC VDD_VALUE

IREF 5 5 DC IREF_VALUE

* PMOS Transistors
MREF 5 X 5 5 PMOS_MODEL
M1 4 X 2 2 PMOS_MODEL

* Connections
ICOPY 4 2 DC ICOPY_VALUE

* Models
.model PMOS_MODEL PMOS(L=1u W=10u)

.end