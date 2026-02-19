plaintext
* SPICE netlist for the schematic

* Voltage Source
V1 2 0 DC 0

* Current Source
I1 1 2 DC I_g

* Capacitors
C_CB 1 4 C_CB
C_EB 2 3 C_EB

* Diodes
D1 4 8 ICR_Model
D2 3 10 IS_BetaF_Model
D3 3 6 IS_BetaF_Model
D4 2 7 IER_Model

* Resistors
R_C 4 C RC
R_E 6 E RE

.model ICR_Model D (n=2)
.model IS_BetaF_Model D (n=1)
.model IER_Model D (n=2)

* Connections are based on red annotations and nodes
.END