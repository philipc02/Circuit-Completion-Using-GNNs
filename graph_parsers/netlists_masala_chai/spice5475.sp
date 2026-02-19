plaintext
* NPN BJT Amplifier Circuit

VCC 3 0 DC 25V

C1 Vin 4 1uF

R1 3 4 47k
R2 4 0 4.7k
RC 3 2 10k
RE 2 0 1k

Q1 3 4 2 QNL

.model QNL NPN (Is=1e-14 Bf=150)

Vin Vin 0

.END