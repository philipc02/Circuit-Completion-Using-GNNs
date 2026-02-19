plaintext
* SPICE Netlist
* Node numbers are derived from the annotated image

* Voltage Source
V1 1 0 DC Vi

* Resistors
R1 1 2 R1
RF 2 3 RF
R3 3 2 R3
R2 2 0 R2
RL 3 0 RL

* Current Source
Io 3 0 DC Io

* Operational Amplifier
* Ideal op-amp model (no parameters specified)
XOPAMP 2 0 3 OPAMP

* End of Netlist