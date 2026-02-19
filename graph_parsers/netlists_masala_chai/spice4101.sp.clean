plaintext
* SPICE Netlist

V1 1 0 AC 1         ; AC input voltage source vi
VDD 6 0 DC 9        ; Positive DC supply
VSS 2 0 DC -9       ; Negative DC supply

RS 6 4 12k          ; RS = 12k Ohms
RG 3 0 50k          ; RG = 50k Ohms
RO 4 vo 1k          ; RO, assume here is 1k since no value specified
RL vo 7 10k         ; RL = 10k Ohms
CC 4 vo 1u          ; CC, assume here is 1uF since no value specified

M1 4 3 2 2 NMOS     ; NMOS Transistor, drain=4, gate=3, source=2, body=2

.end