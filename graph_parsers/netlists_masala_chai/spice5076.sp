spice
* SPICE Netlist for Inverting Amplifier Circuit

* Voltage Source
V1 4 0 AC 1

* Operational Amplifier
* Assumed to be an ideal op-amp for this setup
* Positive input: Node 4 (non-inverting terminal)
* Negative input: Node 2 (inverting terminal)
* Output: Node 2

* Feedback Connection
R1 2 3 1k  ; Example feedback resistor

* Output Node
* Node 2 connects to Vout

* Ground
* Node 0 is the ground reference

.end