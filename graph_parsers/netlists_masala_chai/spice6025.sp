plaintext
* Differential Amplifier Circuit

V1 3 0 DC <vicm_value> ; Define the DC input voltage value as needed
RC1 4 2 <RC_value>     ; Define the resistor value for RC
RC2 6 2 <RC_value>     ; Define the resistor value for RC (duplicate for Q2)
REE 7 0 <REE_value>    ; Define the resistor value for REE

Q1 4 3 7 NPN           ; NPN transistor Q1: C 4, B 3, E 7
Q2 6 5 7 NPN           ; NPN transistor Q2: C 6, B 5, E 7

*.MODEL NPN NPN(<parameters>) ; Specify the parameters for the NPN model

.END