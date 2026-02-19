plaintext
* AC Voltage Source
V1 2 5 AC 1

* Resistors
RS 2 3 10k
RL 3 5 10k

* Diode
D1 3 4 D_1N4148

* Model for Diode
.model D_1N4148 D(Is=1e-14)

* Analysis
.AC LIN 1 1k 1k

.END