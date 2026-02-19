plaintext
* SPICE netlist for the given schematic

Q1 2 4 3 QMOD

RB 7 2 RB_value
RC 4 5 RC_value

VBB 7 6 VBB_value
VCC 5 6 VCC_value

.model QMOD NPN

* Voltage source values, resistor values, and model parameters (if needed) should be defined by the user.