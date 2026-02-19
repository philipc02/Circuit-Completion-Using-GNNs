plaintext
* NPN Transistor Amplifier Circuit

Q1 2 4 3 NPN

RC 1 2 1k ; Resistor RC
RE 3 3 1k ; Resistor RE

VCC 1 0 DC 10 ; Voltage source VCC
VEE 3 0 DC -10 ; Voltage source VEE

* Nodes:
* 1 - Connection to VCC
* 2 - Collector of Q1
* 3 - Emitter of Q1, Connection to VEE
* 4 - Base of Q1, Input node

.model NPN NPN
.end