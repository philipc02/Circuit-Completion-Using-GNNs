plaintext
* Op-Amp Inverting Amplifier

* Voltage Input
Vi 1 0 DC 1

* Resistors
R1 1 2 20k
R2 3 2 100k

* Operational Amplifier
XOP 0 2 3 OPAMP

* Control Statements
.option nomod
.control
run
plot v(3)
.endc

* Subcircuit for Opamp
.subckt OPAMP 1 2 3
E1 3 0 2 1 1E6
.ends OPAMP

.end