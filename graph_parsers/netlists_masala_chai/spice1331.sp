plaintext
* SPICE Netlist for the given schematic

* Voltage Sources
V1 1 0 DC V1_DC
V2 4 0 DC V2_DC

* Resistors
R1 4 3 R1_value
R2 1 3 R2_value
RF 2 3 RF_value

* Operational Amplifier
* (model of the op-amp would be defined separately)
XOPAMP 3 0 2 OPAMP_MODEL

.end