plaintext
* Declare nodes
* 1 - VCC
* 2 - Collector
* 3 - Ground
* 4 - Base node of R1
* 5 - Junction between R1 and R2
* 6 - Emitter node of RE and R2
* 7 - Emitter node of R1

* Voltage Source
VCC 1 3 DC 10

* Resistors
R1 4 5 68k
R2 5 6 36k
RC 1 2 42k
RE 6 3 30k

* NPN Transistor
Q1 2 5 6 NPN

.end