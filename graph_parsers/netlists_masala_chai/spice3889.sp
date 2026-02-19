plaintext
* Circuit using BJT, Resistors, and Voltage Sources

V1 3 0 DC 3V
V2 6 0 DC -3V
V3 2 0 DC -1V

Q1 3 2 5 BJT_MODEL

RB 2 0 500k
RE 5 6 4.8k

.model BJT_MODEL NPN (IS=1e-14 BF=100)

.end