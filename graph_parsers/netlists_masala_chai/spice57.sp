spice
* SPICE Netlist for the Circuit

* Voltage Sources
V1 6 0 DC <Vs_value> ; Input Voltage Source
VCC 5 0 DC <Vcc_value> ; Positive Supply Voltage
VEE 0 2 DC <Vee_value> ; Negative Supply Voltage

* Resistors
RS 6 3 1k ; Source Resistor
RL 1 2 1k ; Load Resistor

* NPN Transistor
Q1 5 3 1 NPN  ; Collector at net 5, Base at net 3, Emitter at net 1

* Specify model for NPN transistor
.model NPN NPN

.end