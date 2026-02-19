spice
* Transistor
Q1 4 2 5 npn

* Resistors
RB 3 2 RB_value
RC 4 4 RC_value

* Voltage Sources
VBB 3 0 VBB_value
VCC 4 5 VCC_value

* Analysis
.dc VBB 0 10 0.1
.end