plaintext
* Circuit Netlist

VDD 5 0 DC 1.8V

* NMOS Transistor: M1 (drain, gate, source, body)
M1 6 3 1 1 NMOS_MODEL

* PMOS Transistor: M2 (drain, gate, source, body)
M2 4 6 5 5 PMOS_MODEL

* Resistors
R1 4 3 10k
R2 6 2 1k

* Voltage Supply
V1 5 0 DC 1.8V

* Models (example placeholder values)
.model NMOS_MODEL nmos (level=1 Vto=0.7)
.model PMOS_MODEL pmos (level=1 Vto=-0.7)

.end