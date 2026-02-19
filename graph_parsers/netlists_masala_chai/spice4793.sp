plaintext
* BJT Circuit
V1 1 0 DC 0

R1 1 4 10k
RC 2 6 8.2k
RE 6 0 1k

D1 4 5 D
D2 5 5 D
D3 5 0 D

Q1 6 4 0 QN

.model D D
.model QN NPN

VCC 2 0 DC 12V