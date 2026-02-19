* SPICE Netlist for the given schematic

* Voltage Source
V1 2 0 DC 0

* Operational Amplifiers
* Ideal Op-Amps are used here
XA1 2 2 V01 opamp
XA2 3 3 V02 opamp

* Resistors
R1 2 3 R1_value
R2 2 4 R2_value
RL 4 5 RL_value

* Voltage Outputs
V1O1 V01 0 DC 0
V1O2 V02 0 DC 0

* Ground connection
VGROUND 3 0 DC 0

.model opamp opamp
.end