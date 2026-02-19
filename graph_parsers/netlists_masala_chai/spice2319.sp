plaintext
* NMOS Transistor (M1)
M1 2 1 0 0 NMOS

* Current Source (IBIAS)
I1 3 2 DC IBIAS

* Voltage Sources (for clarity)
V1 1 0 VIN_DC
V2 3 0 VBIAS

* Model definition for NMOS (example, specific parameters need to be defined based on real components)
.model NMOS NMOS (LEVEL=1)

* Analysis Commands
*.dc VIN_DC 0 5 0.1
*.print DC V(2)
*.end