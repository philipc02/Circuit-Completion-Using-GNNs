spice
* Differential Amplifier Circuit

VCC 9 0 DC 15V
VEE 0 7 DC -15V

RC1 9 4 2k
RC2 9 5 2k
RE 3 7 1k

Q1 4 2 3 NPN
Q2 5 6 3 NPN

* Voltage sources
V1 2 0
V2 6 0

.MODEL NPN NPN (IS=1E-14 BF=100)

.END