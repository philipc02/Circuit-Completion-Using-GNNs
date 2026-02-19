spice
* Op-Amp Circuit

* Voltage Inputs
VCC 2 0 DC 15
VEE 4 0 DC -15
VIN vin 0 DC 0

* Resistors
R1 2 4 1k
R2 3 4 1k

* Capacitor
CBY 3 4 1u

* Op-amp
XU1 3 2 VCC VEE Vout opamp

* Subcircuit for Op-amp
.subckt opamp +in -in v+ v- out
* Idealized op-amp model
.ends opamp

.control
run
.endc