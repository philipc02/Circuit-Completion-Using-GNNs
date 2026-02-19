* SPICE Netlist
* Transistors
QREF 3 3 4 QN
Q1 2 2 0 QN

* Current Source
IREF 3 1 DC

* Resistor
RP 4 0 1k

* Voltage Source
VCC 1 0 DC 5V

* Models
.model QN NPN (IS=1e-14 BF=100)

.end