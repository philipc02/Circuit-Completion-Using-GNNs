spice
* SPICE Netlist

VCC 1 0 DC 15V
VEE 0 7 DC -15V
VIN 8 0 AC 1V

Q1 4 3 5 QMODEL

R1 3 0 2k
R2 3 6 47k
RL 6 5 100

XU1 3 3 4 7 OPAMP_INSTANCE

.model QMODEL NPN(IS=1e-14 BF=100)
.subckt OPAMP_INSTANCE 1 2 3 4
* Subcircuit definition for 741C op-amp
.ends OPAMP_INSTANCE

.end