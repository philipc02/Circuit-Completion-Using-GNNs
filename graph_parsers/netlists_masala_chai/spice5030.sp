plaintext
* BJT Differential Pair Circuit

VCC 3 0 DC
VEE 5 0 DC

V1 4 0 AC
V2 2 0 AC

RC1 3 6 RC
RC2 7 8 RC
RE 5 9 RE

Q1 6 4 5 NPN
Q2 8 2 5 NPN

* Connections: 
* Node 3: VCC
* Node 5: VEE 
* Node 6: Collector of Q1, RC1
* Node 8: Collector of Q2, RC2
* Node 4: Base of Q1, v1
* Node 2: Base of Q2, v2
* Node 9: Connected emitter for both Q1 and Q2 via RE
* Node 0: Ground

.END