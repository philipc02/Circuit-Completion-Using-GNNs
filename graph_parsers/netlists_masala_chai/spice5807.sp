spice
* NPN Transistor Circuit

VBE 3 2 DC 0.7V ; V_BE voltage source
VCC 7 4 DC 12V  ; V_CC voltage source
Rc 4 6 1k       ; Collector resistor

Q1 4 3 2 NPN    ; NPN Transistor

* .model NPN is required if using a simulation

.end