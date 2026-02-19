plaintext
* Op-Amp Inverting Amplifier
* Node List: (1=vi input, 2=op-amp inverting input, 3=op-amp output, 4=ground, 5=vo output)
R1 1 2 20k
R2 4 2 20k
R3 2 3 200k
R4 3 5 200k
XU1 4 2 3 OPAMP
V1 1 4 DC vi

* Ground Definition
VSS 4 0 DC 0

* Model Definition for OPAMP
.model OPAMP opamp