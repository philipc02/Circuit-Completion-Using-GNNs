spice
* SPICE netlist
V1 4 0 DC v1
V2 5 0 DC v2

R1 4 1 R
R2 1 2 R
R3 5 6 R
R4 6 2 R

* Op-Amp
* Assuming ideal op-amp
XOPAMP 2 7 3 OPAMP

.MODEL OPAMP OPAMP

.END