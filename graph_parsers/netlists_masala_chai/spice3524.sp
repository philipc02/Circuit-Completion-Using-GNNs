spice
* Transistor connections
* Qx model is assumed to be NPN

Q1 2 8 3 NPN
Q2 3 6 4 NPN
Q3 10 8 2 NPN
Q4 9 3 5 NPN
Q5 2 2 3 NPN
Q6 2 4 3 NPN

* Voltage sources
V1 6 0 +VCC
V2 10 0 -VEE

* Resistors
RC1 6 3 RC1_value
RC2 3 4 RC2_value
RC3 4 7 RC3_value
RE1 3 0 RE1_value
RE2 4 0 RE2_value
RE3 5 0 RE3_value
RB1 2 8 RB1_value
RB2 4 9 RB2_value
RL 7 9 RL_value
RS 8 0 RS_value
RE4 9 0 RE4_value

* Output
VOUT 7 0 DC 0

.end