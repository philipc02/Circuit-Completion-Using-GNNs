plaintext
* SPICE Netlist for the given schematic

VCC 1 0 DC 2.5V     ; VCC = 2.5V
I1 2 3 DC 1mA       ; Current source 1 mA

R1 4 0 1k           ; R1, assumed to be 1k for example
RC 1 2 1k           ; RC, assumed to be 1k for example
RF 3 3 1k           ; RF, assumed to be 1k for example (Note: RF is connected in parallel with itself, it's equivalent to short-circuit its value doesn't matter)

C1 4 5 100p         ; Input capacitor 100 pF
CL 2 0 1u           ; Load capacitor CL, assumed to be 1u for example

Q1 2 4 3 NMOS       ; NMOS transistor Q1, drain=2, gate=4, source=3

* Nodes:
* 1 - VCC, top of RC
* 2 - Collector of Q1, one side of CL
* 3 - Common ground
* 4 - Input, connected to R1 and gate of Q1
* 5 - Output side of C1

.END