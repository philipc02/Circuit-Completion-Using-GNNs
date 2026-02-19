spice
* NPN Transistor Circuit
V1 6 0 DC 0 ; Voltage source
RS 6 2 1000 ; Resistor RS with assumed value 1k ohms
RL 5 7 1000 ; Resistor RL with assumed value 1k ohms
Q1 3 2 5 NPN ; NPN transistor with collector at net 3, base at net 2, emitter at net 5
VCC 3 0 DC 10 ; Power supply for the collector

* Input
VI 6 0 DC 5 ; Input voltage source

* Output
VO 5 4 DC 0 ; Output voltage

.END