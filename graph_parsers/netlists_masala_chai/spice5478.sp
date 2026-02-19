plaintext
* SPICE Netlist for BJT Amplifier Circuit

VCC 3 0 DC VCC
R1 3 5 R1
R2 5 0 R2
RE 2 0 RE
C1 1 5 C1
Q1 3 5 2 NPN

.model NPN NPN (IS=1e-14 BF=100)

* Connections:
* Node 3: +VCC Connection
* Node 5: Base of the BJT
* Node 2: Emitter of the BJT connected to RE
* VCC connected to collector via node 3
* Capacitor C1 connected to input at node 1 and base at node 5
*
.ends