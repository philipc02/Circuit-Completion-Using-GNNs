plaintext
* SPICE Netlist for the given circuit

* Resistors
R1 1 2 R1_value
R2 3 4 R2_value

* Voltage Input
V1 vi 1 DC vi_value

* Operational Amplifier
* Positive input at node 2, Negative input at node 2, Output at node 3
X1 2 2 3 opamp_model

* Ground
V2 2 0 DC 0

* Output voltage
Vout vo 3 DC 0

.end