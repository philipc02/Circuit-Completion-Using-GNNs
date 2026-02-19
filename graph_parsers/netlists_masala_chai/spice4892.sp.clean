plaintext
** Netlist for the given schematic **
VCC 4 0 DC <value>    ; Define VCC value
R1 4 3 <value>        ; Connects VCC to the base of the BJT
R2 3 0 <value>        ; Connects the base of the BJT to ground
RE 2 0 <value>        ; Connects the emitter of the BJT to ground

C1 1 3 <value>        ; Input coupling capacitor
C2 2 0 <value>        ; Emitter bypass capacitor
C3 6 7 <value>        ; Capacitor in LC tank circuit

L1 6 7 <value>        ; Inductor in LC tank circuit

Q1 5 3 2 NPN          ; NPN BJT with collector at 5, base at 3, and emitter at 2

* Define nodes:
* 1 - Input
* 2 - Emitter
* 3 - Base
* 4 - VCC
* 5 - Collector / Output to next stage
* 6, 7 - LC tank circuit

.END