plaintext
* BJT Amplifier Circuit
RB 2 1 <RB_value>
Q1 1 2 3 NPN
I1 2 0 DC <I_value>
V1 3 0 AC <ac_value>
* Additional SPICE commands and definitions
.model NPN NPN(IS=1e-15 BF=100)