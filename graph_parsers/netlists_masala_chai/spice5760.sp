spice
* SPICE Netlist
V1 3 0 DC 5        ; Voltage source: 5V connected to node 3
R1 7 4 200k        ; Resistor: 200k ohm connected from node 7 to ground
R2 2 4 2k          ; Resistor: 2k ohm connected from node 2 to ground
Q1 5 7 2 QNPN      ; NPN Transistor: Collector 5, Base 7, Emitter 2

.model QNPN NPN    ; Define NPN model (example parameters)