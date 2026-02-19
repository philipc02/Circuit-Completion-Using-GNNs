spice
* SPICE Netlist for the given op-amp circuit

R1 1 2 R
R2 2 4 R
R3 4 0 R
R4 3 0 R
R5 2 5 R5
R6 5 0 R6

* Operational Amplifier
XOPAMP 3 4 5 OPAMP_MODEL

V1 1 0 dc v1_value
V2 3 0 dc v2_value

* Specify op-amp model
.model OPAMP_MODEL OPAMP