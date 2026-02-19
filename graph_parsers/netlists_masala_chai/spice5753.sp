spice
* SPICE Netlist for the circuit

V1 5 0 DC 15V      ; Voltage source

Q3 5 6 2 QNPN      ; NPN Transistor (Collector, Base, Emitter)

R1 2 3 470         ; Resistor 470Ω

.model QNPN NPN    ; Model for NPN

.END