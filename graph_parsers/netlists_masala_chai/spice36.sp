spice
* SPICE netlist for the circuit

* Voltage Source
V1 8 9 DC 0

* Resistors
RS 8 2 20k
Rpi 2 4 26k
RC 5 4 5k
RL 6 5 10k

* Current Source (Voltage Controlled Current Source)
G1 5 4 2 4 1/260

* Ground node
V0 3 0 DC 0

* End of netlist