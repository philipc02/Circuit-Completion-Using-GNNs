* Example SPICE netlist

* NPN Transistor
Q1 3 4 0 NPN

* Current Source
Iin 5 0 DC [value]

* Resistors
RF 4 3 [value]
RC 2 3 [value]

* Voltage Source
Vcc 2 0 DC [value]

* Connections
Vout 3 0

* Model Definitions
.model NPN NPN
.end