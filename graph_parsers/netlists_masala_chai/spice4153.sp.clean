plaintext
* SPICE Netlist for the given schematic

V1 7 8 DC <Voltage_Value> ; Replace <Voltage_Value> with desired voltage

Q1 3 2 4 NPN
Q2 9 4 8 NPN
Q3 3 5 7 PNP
Q4 9 5 7 PNP

R1 5 7 <R1_Value>    ; Replace <R1_Value>
R2 6 8 <R2_Value>    ; Replace <R2_Value>
R3 5 9 <R3_Value>    ; Replace <R3_Value>
R4 8 9 <R4_Value>    ; Replace <R4_Value>
RL 9 0 <RL_Value>    ; Replace <RL_Value>

* Definitions for transistors
.model NPN NPN (BF=100)
.model PNP PNP (BF=100)

.END