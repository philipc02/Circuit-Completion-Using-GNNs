plaintext
* SPICE Netlist for the given Schematic

* Voltage Sources
V1 1 0 DC 5V
V2 3 0 DC -5V

* Current Source
IREF 1 2 DC

* Resistors
R1 1 2 R1_value
RE2 5 3 RE2_value
RE3 4 3 RE3_value

* Transistors
Q1 2 2 3 NPN
Q2 5 6 3 NPN
Q3 4 6 3 NPN

* Current sources for outputs
IO2 6 5 DC
IO3 6 4 DC

.end