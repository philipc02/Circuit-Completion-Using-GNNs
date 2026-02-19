plaintext
* NPN Transistor Amplifier Circuit

VCC 12 0 DC 10
Vin 4 0 AC 2mV

Q1 2 3 5 QNPN

R1 12 3 10k
R2 3 4 2.2k
RC 12 2 3.6k
RE 5 0 1k

C1 2 0 1u
C2 11 0 1u

.model QNPN NPN (IS=1E-14 BF=100)

.end