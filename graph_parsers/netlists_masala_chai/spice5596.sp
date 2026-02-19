spice
* Op-Amp Circuit
R1 1 3 1k
R2 3 2 1k
C1 2 3 1uF
V1 1 0 DC 5V
XU1 3 0 3 OPAMP_MODEL

* Subcircuit for Op-Amp Model
.subckt OPAMP_MODEL non_inv inv out
* Op-Amp model goes here
.ends OPAMP_MODEL

.end