spice
* Differential Pair Circuit

VCC 5 0 DC VCC

RC1 5 1 RC
RC2 5 2 RC

Q1 1 3 3 NPN
Q2 2 4 3 NPN

RE 3 0 RE

IEE1 3 0 DC IEE
IEE2 4 0 DC IEE

VIN1 3 6 DC Vin1
VIN2 4 7 DC Vin2

* Node Mapping:
* 0 - Ground
* 1 - Collector of Q1
* 2 - Collector of Q2
* 3 - Emitters of Q1, Q2 and IEE connections
* 4 - Base of Q2
* 5 - VCC
* 6 - Input Vin1
* 7 - Input Vin2

.model NPN NPN