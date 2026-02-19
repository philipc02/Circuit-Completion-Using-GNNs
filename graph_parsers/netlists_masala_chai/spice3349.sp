plaintext
* BJT and Diode Circuit
Q1 4 2 3 BJTModel
D1 2 3 DiodeModel

* Models
.model BJTModel NPN (BF=100)
.model DiodeModel D

* Node Voltage Sources for reference
V1 4 0 DC 0
V2 2 0 DC 0
V3 3 0 DC 0