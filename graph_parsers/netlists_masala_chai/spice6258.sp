plaintext
* BJT Differential Pair with Current Mirror

VCC 2 0 DC VCC
VBIAS 1 0 DC VBIAS
I1 2 0 DC I

Q1 3 1 4 QNPN
Q2 3 5 3 QNPN
Q5 2 2 3 QNPN

RC1 4 0 RC
RC2 3 0 RC

.model QNPN NPN (IS=1E-15 BF=100)
.end