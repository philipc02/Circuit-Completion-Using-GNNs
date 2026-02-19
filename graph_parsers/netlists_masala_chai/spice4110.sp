spice
* Voltage source
V1 5 0 DC 0.0V

* Current source
I1 2 6 DC 0.2mA

* Resistors
R1 5 2 10k
RC 7 2 10k
RL 3 4 20k

* Capacitors
CC 2 3 1u
CE 2 6 1u

* Voltage input
V+ 1 7 DC 3V
V- 6 0 DC -3V

* Connections for testing purposes
VTEST vi 5 0 DC 0V
VOUT vo 4 0
.end