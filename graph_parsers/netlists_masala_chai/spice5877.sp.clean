spice
* Transistor
Q1 4 2 5 NPN

* Resistors
R1 6 2 R1_value
R2 2 3 R2_value
RE 4 5 RE_value

* Current Source
IO 4 5 DC I0_value

* Voltage Source
VCC 6 3 DC VCC_value

* Simulation Control
.control
  dc VCC 0 10 0.1
  print V(2) V(4) I(VCC)
.endc

.model NPN NPN
.END