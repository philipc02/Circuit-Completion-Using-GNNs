plaintext
* Node Definitions:
* 0 = Ground
* 1 = Node at +10V
* 2 = Node between D2 and R1
* 3 = Node between R1 and R2
* 4 = Node between R2 and R3
* 5 = Node at -5V

V1 1 0 DC 10
V2 2 0 DC 5
V3 5 0 DC -5

D1 1 2 Dmodel
D2 2 3 Dmodel
D3 0 4 Dmodel

R1 2 3 1k
R2 3 4 1k
R3 4 5 1k

.model Dmodel D
.END