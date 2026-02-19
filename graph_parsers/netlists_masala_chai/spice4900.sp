plaintext
* BJT Amplifier Circuit
VCC 7 0 DC VCC
Vin 5 0 AC Vin

R1 7 2 R1
R2 2 5 R2
RE 8 0 RE
RL 3 0 RL

Q1 4 2 8 QNPN

C1 5 2 C1
C2 4 3 C2
C3 4 0 C3

.model QNPN NPN (IS=1e-14 BF=100 NF=1 VAF=100)