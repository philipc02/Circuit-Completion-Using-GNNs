plaintext
* SPICE Netlist for the given schematic

VDD 1 0 DC 5V
Vb 6 9 DC 1.2V 

* PMOS Transistors
M1 3 2 1 1 PMOS
M2 4 2 1 1 PMOS

* Resistors
RP1 11 3 1k
RP2 5 2 1k
RP3 7 4 1k

* Current Sources
I1 6 9 DC 50uA
I2 8 6 DC 50uA

* Voltage source node definition (assuming VDD on source)
V1 0 1 DC

.model PMOS PMOS(Level=1)
.end