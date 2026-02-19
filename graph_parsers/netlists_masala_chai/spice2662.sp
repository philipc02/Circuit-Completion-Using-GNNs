plaintext
* Current Source
IREF 7 4 DC IREF

* Voltage Source
VDD 8 0 DC VDD

* PMOS Transistors
* Left W_r PMOS
W_r_left 4 4 7 PMOS_MODEL

* Right W_r PMOS
W_r_right 4 4 7 PMOS_MODEL

* Left W_d PMOS
W_d_left 3 3 4 PMOS_MODEL

* Right W_d PMOS
W_d_right 3 3 4 PMOS_MODEL

* NMOS Transistors
* M1 NMOS
M1 6 4 0 NMOS_MODEL

* M2 NMOS
M2 2 3 0 NMOS_MODEL

* Models (assuming generic models are defined)
.model PMOS_MODEL PMOS
.model NMOS_MODEL NMOS