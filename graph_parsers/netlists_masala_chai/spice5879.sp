* NMOS configuration with gate, source, and drain nodes labelled.
M1 2 5 3 3 NMOS

* Resistor connected between +5V and node 2.
R1 1 2 R

* Voltage source providing +5V.
V1 1 0 DC 5

* Current source flowing from node 3 to ground.
I1 3 4 DC Io

* Ground connection.
V2 4 0 DC 0

.model NMOS NMOS
.end