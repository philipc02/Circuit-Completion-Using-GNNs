plaintext
* NPN Transistor Circuit
V1 1 0 DC 12V

R1 2 0 620
RC 1 3 200

Q1 3 2 0 NPN

D1 3 2 DZENER
D2 0 3 DLED

.model DZENER D BV=6.2V
.model DLED D

.end