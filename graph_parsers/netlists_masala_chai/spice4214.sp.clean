plaintext
* SPICE Netlist
V1 1 0 DC v1/1
V2 2 0 DC v2/2

* Operational Amplifier 1
* Positive input at node 1, negative input at node 3, output at node 4
XU1 1 3 v01 OpAmp

* Operational Amplifier 2
* Positive input at node 2, negative input at node 2, output at node v02
XU2 2 2 v02 OpAmp

R1 2 3 R
RL 3 4 RL

* Model Definitions
.model OpAmp opamp

.end