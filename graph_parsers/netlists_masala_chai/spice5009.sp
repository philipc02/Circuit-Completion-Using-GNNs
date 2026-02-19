spice
* SPICE Netlist for Provided Schematic

VBB 2 0 DC [VBB_VALUE]

R1 2 6 [R1_VALUE]
R2 6 7 [R2_VALUE]
R3 2 3 [R3_VALUE]
R4 5 4 [R4_VALUE]

C1 7 0 [C_VALUE]

Q1 3 1 5 NPN

* Specify the model for the NPN transistor
.model NPN NPN(IS=[IS_VALUE] BF=[BF_VALUE])

.end