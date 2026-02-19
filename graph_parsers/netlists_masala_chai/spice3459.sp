plaintext
* Differential Amplifier Circuit

VCC 4 0 DC VCC
V11 7 0 DC V11
V12 8 0 DC V12

RC1 4 6 RC
RC2 4 6 RC
RE1 5 0 RE
RE2 5 0 RE
RL1 6 2 RL
RL2 6 3 RL

Q1 6 7 5 NPN
Q2 6 8 5 NPN

IBIAS 5 0 DC IBIAS

.model NPN NPN (IS=1e-15 BF=100)

.end