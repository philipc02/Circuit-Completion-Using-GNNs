* Netlist for the given circuit

* Voltage Source
VREF 4 2 DC <VREF_value>

* Resistors
R1 2 2 <R1_value>
R2 2 2 <R2_value>
RP 3 4 <RP_value>  ; RP is the parallel combination R1||R2

* Op-Amp
XOP 3 22 2 OPAMP ; Custom Op-Amp model

* Nodes
* 2 - Common node linked to VREF, v+, v-, and Op-Amp Input.
* 3 - Node for v+ input to Op-Amp from RP
* 4 - Node connected to VREF
* 22 - Node connected to v1 and v- input of Op-Amp

.end