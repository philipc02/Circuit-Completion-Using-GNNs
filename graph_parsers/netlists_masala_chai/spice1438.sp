plaintext
* NPN Transistor Amplifier Circuit

VCC Vcc 0 DC 15
Vin Vin 0 AC 1

Q2 Vout Vin 0 NPN

RE Vcc Vout 22k
RC Vout 0 4.7k

.model NPN NPN (BF=100)

.end