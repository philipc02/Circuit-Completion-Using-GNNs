spice
* SPICE netlist for the given schematic

V1 1 7 DC Vin
V2 5 7 DC Vref
R1 1 2 R
R2 2 3 R
XOPAMP 2 7 2 OPAMP
XAMULT 3 5 3 2 AMULT
VOUT 6 7

* Models and Subcircuits
.subckt OPAMP noninv inv out
* Define operational amplifier model here
.ends OPAMP

.subckt AMULT Z Y X out
* Define analog multiplier model here
.ends AMULT

.end