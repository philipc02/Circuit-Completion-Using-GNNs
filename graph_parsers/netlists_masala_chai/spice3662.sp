spice
* Diode Circuit
V1 7 0 DC 15
V2 4 0 DC -10
V3 5 0 DC -5

R1 7 6 6.15k
R2 3 0 2k
R3 5 6 R3_value
R4 4 2 R4_value

D1 6 3 Dmodel
D2 6 5 Dmodel
D3 2 4 Dmodel

.model Dmodel D
.END