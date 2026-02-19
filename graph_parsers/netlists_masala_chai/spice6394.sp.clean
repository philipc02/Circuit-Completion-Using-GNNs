plaintext
* Components
R1 2 2_ground R1_value
R2 3 2 R2_value
R3 4 5 R_value
C1 2_ground 4 C_value
C2 5 8 C_value
C3 6 3 C3_value

* Subcircuit for the Op-Amp
.subckt opamp 2 3 6 5
* (Inputs: in+, in-, out, V+)
Vout 6 7 0
Vin+ 2 5 8
Vin- 3 0 0
Rin 2 6 10MEG
E1 7 0 6 3 1E6
G1 7 0 2 3 GM
.ends opamp

* Connections
XU1 2 3 6 5 opamp

* Power Supplies
V+ 8 0 DC Vcc

* Ground
2_ground 0 0V

* Analysis
.tran 0.1m 10m
.ac dec 10 1 1meg
.end