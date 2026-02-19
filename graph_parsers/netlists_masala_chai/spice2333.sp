spice
* NMOS Transistors
M1 6 1 5 5 NMOS
M2 3 4 5 5 NMOS

* Current Source
I_BIAS 2 0 DC VALUE

* Node Assignments
* 1 = V_IN+
* 3 = I_OUT+
* 4 = V_IN-
* 5 = V_IN,DM
* 6 = I_OUT-

* Example NMOS Model
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=120µ)