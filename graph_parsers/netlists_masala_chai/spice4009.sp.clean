plaintext
* SPICE netlist for the given circuit
VCC V+ 0 DC 5V
VEE V- 0 DC -5V
IS 5 0 DC VS
IE0 3 V+ DC 0.5mA

* Capacitors
CC1 5 0 CC1_value
CC2 3 VO CC2_value

* Resistors
RB 5 0 10k
RL VO 0 0.5k

* Transistor
Q1 3 5 4 QMODEL

* Model definitions
.model QMODEL NPN

.end