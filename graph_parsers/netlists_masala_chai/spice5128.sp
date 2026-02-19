plaintext
* SPICE Netlist for the given schematic

* Voltage sources
V1 3 0 AC 2m
VCC 5 0 DC 15
VEE 6 0 DC -15

* Resistors
RG 3 2 600
R1 2 4 3.3k
R2 2 0 100k
Rf 2 4 150k
RL 2 0 10k

* Capacitors
C1 2 0 1u
C2 2 0 10u
C3 4 0 4.7u

* Operational Amplifier Model
* (This is a placeholder; the actual op-amp model or parameters may vary.)
.model OPAMP741 opamp
X1 2 4 0 OPAMP741

.end