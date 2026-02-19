plaintext
* SPICE Netlist for the given schematic

V1 1 4 DC 0 ; Voltage source vs
RS 4 3 100k ; Resistor RS
RC 2 V+ 10k ; Resistor RC
RB 3 6 100k ; Resistor RB
RE 6 V- 10k ; Resistor RE
Ro 5 vo 1k ; Resistor Ro
RL vo 0 1k ; Resistor RL
CC1 3 7 1u ; Capacitor CC1
C 5 2 1u ; Capacitor C
CC2 5 io 1u ; Capacitor CC2

Q1 2 3 6 QNPN ; NPN Transistor Q1

* Models
.model QNPN NPN

.end