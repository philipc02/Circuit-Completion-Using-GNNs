* Netlist for the given schematic

VCC 5 0 DC 5V
VEE 9 0 DC -5V
I 7 8 DC 1A

* Transistor connections
* Assuming all are BJTs with notation QNAME collector base emitter
Q1 3 6 8 NPN
Q2 7 6 8 NPN
Q3 5 3 22 NPN
Q4 4 22 7 NPN
Q5 5 3 3 NPN
Q6 3 2b 3 NPN
Q7 4 2 7 NPN

* Connections
Vd 2 0
Vbias 2b 2

* Output
Vout 4 0

.END