spice
* SPICE netlist for the provided circuit

V1 6 0 DC Vin
RS 6 4 1k
R3 6 5 10k

Q1 5 5 2 NPN

R2 5 2 5k
R1 2 0 10k
RL 4 2 10k

VZ 2 3 DC 5V

* NPN transistor: collector, base, emitter
.model NPN NPN

.end